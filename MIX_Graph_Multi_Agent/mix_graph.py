import random
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import time
from MIX_Graph_Multi_Agent.replay_buffer import ReplayBuffer
from  MIX_Graph_Multi_Agent.networks import Actor_Soft_Attention, Critic_Soft_Attention, Actor_Attention, Critic_Attention, Actor_full_connect, Critic_full_connect
tf.keras.backend.set_floatx('float32')

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#update network parameters
def update_target_weights(model, target_model, tau):
    weights = model.get_weights()
    target_weights = target_model.get_weights()
    for i in range(len(target_weights)):  # set tau% of target model to be new weights
        target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
    target_model.set_weights(target_weights)

#MADDPGAgent Class
class MIX_GRAPH():
    def __init__(self, name, obs_shape, act_space, agent_num, parameters, create_summary_writer=False):
        self.name = name
        self.obs_shape = obs_shape
        self.act_space = act_space
        self.agent_num = agent_num
        self.group_num = 2
        self.agent_group_index = [0, 0, 0, 0, 1, 1, 1, 1]
        self.parameters = parameters

        self.actors = [Actor_Soft_Attention() for i in range(self.group_num)]
        self.critics = [Critic_Soft_Attention() for i in range(self.group_num)]
        self.actor_targets = [Actor_Soft_Attention() for i in range(self.group_num)]
        self.critic_targets = [Critic_Soft_Attention() for i in range(self.group_num)]

        self.actor_optimizers = [Adam(learning_rate=parameters['lr_actor']) for i in range(self.group_num)]
        self.critic_optimizers = [Adam(learning_rate=parameters['lr_critic']) for i in range(self.group_num)]
        #self.critic.compile(loss="mean_squared_error", optimizer=self.critic_optimizer)

        self.action_noises = [OrnsteinUhlenbeckActionNoise(mu=np.zeros(act_space[i].shape[0]), sigma=parameters['sigma']) for i in range(self.agent_num)]

        self.update_networks_weight(tau=1)

        # Create experience buffer
        self.replay_buffers = [ReplayBuffer(parameters["buffer_size"]) for i in range(self.agent_num)]
        self.max_replay_buffer_len = parameters['max_replay_buffer_len']
        self.replay_sample_index = None

        # 为每一个agent构建tensorboard可视化训练过程
        if create_summary_writer:
            self.summary_writers = []
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            for i in range(self.agent_num):
                train_log_dir = 'logs/MIX_GRAPH_Summary_' + current_time + "agent" + str(i)
                self.summary_writers.append(tf.summary.create_file_writer(train_log_dir))

    def update_networks_weight(self, tau=1):
        for i in range(self.group_num):
            update_target_weights(model=self.actors[i], target_model=self.actor_targets[i], tau=tau)
            update_target_weights(model=self.critics[i], target_model=self.critic_targets[i], tau=tau)

    @tf.function(experimental_relax_shapes=True)
    def action(self, obs_n, evaluation=False):
        action_n = []
        for i, obs in enumerate(obs_n):
            obs = tf.expand_dims(obs, axis=0)
            mu = self.actors[self.agent_group_index[i]](obs)
            noise = np.asarray([self.action_noises[self.agent_group_index[i]]() for j in range(mu.shape[0])])
            # print(noise)
            pi = tf.clip_by_value(mu + noise, -1, 1)
            a = mu if evaluation else pi
            action_n.append(a[0])
        return action_n

    def experience(self, obs_n, action_n, rew_n, new_obs_n, done_n):
        # Store transition in the replay buffer.
        for i in range(self.agent_num):
            self.replay_buffers[i].add(obs_n[i], action_n[i], [rew_n[i]], new_obs_n[i], [float(done_n[i])])

    #save_model("models/maddpg_actor_agent_", "models/maddpg_critic_agent_")
    def save_model(self, a_fn, c_fn):
        for i in range(self.group_num):
            self.actors[i].save_weights(a_fn+str(i)+".h5")
            self.critics[i].save_weights(c_fn+str(i)+".h5")

    def load_actor(self, a_fn):
        for i in range(self.group_num):
            self.actors[i].load_weights(a_fn+str(i)+".h5")

    def preupdate(self):
        self.replay_sample_index = None


    def update(self, train_step):
        T = time.time()
        self.replay_sample_index = self.replay_buffers[0].make_index(self.parameters['batch-size'])
        # collect replay sample from all agents
        obs_n = []
        act_n = []
        obs_next_n = []
        rew_n = []
        done_n = []

        for i in range(self.agent_num):
            obs, act, rew, obs_next, done = self.replay_buffers[i].sample_index(self.replay_sample_index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
            done_n.append(done)
            rew_n.append(rew)

        summaries = self.train((obs_n, act_n, rew_n, obs_next_n, done_n))

        self.update_networks_weight(tau=self.parameters["tau"])

        for i in range(self.agent_num):
            with self.summary_writers[i].as_default():
                for key in summaries.keys():
                    tf.summary.scalar(key, summaries[key][self.agent_group_index[i]], step=train_step)
            self.summary_writers[i].flush()


    @tf.function(experimental_relax_shapes=True)
    def train(self, memories):
        obs_n, act_n, rew_n, obs_next_n, done_n = memories
        act_next_n = []
        for i in range(self.agent_num):
            act_next_n.append(self.actor_targets[self.agent_group_index[i]](obs_next_n[i]))
        with tf.GradientTape(persistent=True) as tape:
            q_loss = [0 for j in range(self.group_num)]
            actor_loss = [0 for j in range(self.group_num)]
            for i in range(self.agent_num):
                # compute Q_target
                q_target = self.critic_targets[self.agent_group_index[i]]([obs_next_n[i]] + act_next_n)
                dc_r = rew_n[i] + self.parameters['gamma'] * q_target * (1 - done_n[i])
                # compute Q
                q = self.critics[self.agent_group_index[i]]([obs_n[i]] + act_n)
                td_error = q - dc_r
                q_loss[self.agent_group_index[i]] += tf.reduce_mean(tf.square(td_error))

                mu = self.actors[self.agent_group_index[i]](obs_n[i]) #batch_size, 2
                act_agents = [mu if i == j else act for j, act in enumerate(act_n)]
                q_actor = self.critics[self.agent_group_index[i]]([obs_n[i]] + act_agents)
                actor_loss[self.agent_group_index[i]] += -tf.reduce_mean(q_actor)
        for i in range(self.group_num):
            q_grads = tape.gradient(q_loss[i], self.critics[i].trainable_variables)
            self.critic_optimizers[i].apply_gradients(zip(q_grads, self.critics[i].trainable_variables))
            actor_grads = tape.gradient(actor_loss[i], self.actors[i].trainable_variables)
            self.actor_optimizers[i].apply_gradients(zip(actor_grads, self.actors[i].trainable_variables))

        summaries = dict([
            ['LOSS/actor_loss', actor_loss],
            ['LOSS/q_loss', q_loss],
        ])

        return summaries


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise(object):
    def __init__(self, mu, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

