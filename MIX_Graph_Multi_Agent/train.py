import numpy as np
import tensorflow as tf
import time
from MIX_Graph_Multi_Agent.mix_graph import MIX_GRAPH
from MIX_Graph_Multi_Agent.observation_process import obs_process
from MIX_Graph_Multi_Agent.buffer import ReplayBuffer
train_parameters = {
    "max-episode-len":25,
    "num-episodes":100000,
    }
model_parameters = {
    "buffer_size":100000,
    "lr_actor":1.0e-2,
    "lr_critic":1.0e-2,
    "sigma":0.15,
    "gamma":0.95,
    "batch-size":1024,
    "max_replay_buffer_len": 10240,
    "tau":0.01
}
from make_env import make_env
def train():
    #初始化环境
    env = make_env(scenario_name="Predator_prey_4v4", discrete_action_space=False, discrete_action_input = False)
    #group_num = 2
    #replay_buffers = [ReplayBuffer(model_parameters["buffer_size"]) for i in range(env.n)]
    #初始化MADDPGAgent
    mix_graph_agents = MIX_GRAPH(name="Predator_prey_4v4",
                                obs_shape=env.observation_space,
                                act_space=env.action_space,
                                agent_num=env.n,
                                parameters=model_parameters,
                                create_summary_writer=True)

    print('Starting training...')
    episode = 0
    epoch = 0
    train_step = 0
    while episode < train_parameters["num-episodes"]:
        t_start = time.time()
        obs_n = env.reset()
        obs_n = obs_process(obs_n)
        episode_steps = 0
        episode_rewards = [0.0]  # sum of rewards for all agents
        group_rewards = [[0.0], [0.0]]
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward

        for action_noise in mix_graph_agents.action_noises:
            action_noise.reset()

        while episode_steps < train_parameters["max-episode-len"]:
            T = time.time()
            # get action
            action_n = mix_graph_agents.action(obs_n, evaluation=False)
            action_n = [np.array(action) for action in action_n]
            # environment step
            #print(action_n)
            new_obs_n, rew_n, done_n, _ = env.step(action_n)
            new_obs_n = obs_process(new_obs_n)
            #env.render()
            done = all(done_n)
            # collect experience
            mix_graph_agents.experience(obs_n, action_n, rew_n, new_obs_n, done_n)

            #记录reward数据
            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew
            group_rewards[0][-1] += sum(rew_n[:4])
            group_rewards[1][-1] += sum(rew_n[4:])

            # update all trainers
            if epoch % 50 == 0:
                if len(mix_graph_agents.replay_buffers[0]) > mix_graph_agents.max_replay_buffer_len:
                    mix_graph_agents.update(train_step)
                    train_step += 1

            if done:
                break

            obs_n = new_obs_n
            episode_steps += 1
            epoch += 1

            #print(time.time()-T)

        print("episode {}: {} total reward, {} epoch, {} episode_steps, {} train_steps, {} time".format(
            episode, agent_rewards, epoch, episode_steps, train_step, time.time()-t_start))

        episode += 1

        for i, summary_writer in enumerate(mix_graph_agents.summary_writers):
            with summary_writer.as_default():
                tf.summary.scalar('Main/total_reward', episode_rewards[-1], step=episode)
                tf.summary.scalar('Main/Agent_reward', agent_rewards[i][-1], step=episode)
                if i < 4:
                    tf.summary.scalar('Main/group_reward', group_rewards[0][-1], step=episode)
                else:
                    tf.summary.scalar('Main/group_reward', group_rewards[1][-1], step=episode)
            summary_writer.flush()

        if episode % 20 ==0:
            # 保存模型参数
            mix_graph_agents.save_model("models/MIX_GRAPH_actor_agent_", "models/MIX_GRAPH_critic_agent_")

    #关闭summary，回收资源
    for i, mix_graph_agent in enumerate(mix_graph_agents):
        mix_graph_agent.summary_writer.close()
    env.close()
    # 保存模型参数
    mix_graph_agents.save_model("models/MIX_GRAPH_actor_agent_", "models/MIX_GRAPH_critic_agent_")

def inference(episode_num=100, max_episode_steps=100):
    # 初始化环境
    env = make_env(scenario_name="Predator_prey_4v4", discrete_action_space=False, discrete_action_input = False)
    # 初始化MADDPGAgent
    mix_graph_agents = MIX_GRAPH(name="Predator_prey_4v4",
                            obs_shape=env.observation_space,
                            act_space=env.action_space,
                            agent_num=env.n,
                            parameters=model_parameters,
                            create_summary_writer=False)

    mix_graph_agents.load_actor(a_fn="models/MIX_GRAPH_actor_agent_")

    episode = 0
    while episode < episode_num:
        rewards = np.zeros(env.n, dtype=np.float32)
        cur_state = env.reset()
        cur_state = obs_process(cur_state)
        step = 0
        while step < max_episode_steps:
            # get action
            action_n = mix_graph_agents.action(cur_state, evaluation=True)
            action_n = [np.array(action) for action in action_n]
            #print(action_n)
            # environment step
            next_state, reward, done, _ = env.step(action_n)
            next_state = obs_process(next_state)
            env.render()
            time.sleep(0.02)
            cur_state = next_state
            rewards += np.asarray(reward, dtype=np.float32)
            step += 1
        episode += 1
        print("episode {}: {} total reward, {} steps".format(
            episode, rewards, step))
    env.close()

if __name__ == '__main__':
    train()
    inference()