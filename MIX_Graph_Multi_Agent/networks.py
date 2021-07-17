import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate, BatchNormalization, LeakyReLU, Bidirectional, LSTM
from MIX_Graph_Multi_Agent.Attention.multi_head_attention import MultiHeadAttention
tf.keras.backend.set_floatx('float32')

def Actor_Soft_Attention(hidden_dim=32):
    input = Input(shape=(8, 36))  # (batch_size, num_agents, obs_length)

    local_inputs = tf.gather(input, axis=1, indices=[0, 1, 2])# (batch_size, 3, obs_length)
    agent, neighbor_agents = tf.split(local_inputs, axis=1, num_or_size_splits=[1, 2])  # (batch_size, 1, obs_length), (batch_size, 2, obs_length)
    agent = tf.tile(agent, [1, 2, 1])# (batch_size, 2, obs_length)
    agents = tf.concat([agent, neighbor_agents], axis=-1)# (batch_size, 2, obs_length*2)

    h_ij_1 = Dense(hidden_dim, activation='relu')(agents)  # batch_size, 2, hidden_dim
    e_ij_1 = Dense(hidden_dim, activation='relu')(agents)  # batch_size, 2, hidden_dim
    a_ij_1 = tf.nn.softmax(e_ij_1, axis=1)  # batch_size, 2, hidden_dim
    h_i_1 = tf.reduce_sum(a_ij_1 * h_ij_1, axis=1)  # batch_size, hidden_dim

    xxx = tf.concat([h_i_1, tf.reshape(local_inputs, [-1, 3*36])], axis=-1)

    xxx = Dense(hidden_dim, activation='relu')(xxx)
    output = Dense(2, activation='tanh')(xxx)  # batch_size, 2
    model = Model(inputs=input, outputs=output)

    return model

def Critic_Soft_Attention(hidden_dim=32):
    input = Input(shape=(8, 36))  # (batch_size, num_agents, obs_length)
    actions = [Input(shape=(2)) for i in range(8)]

    agent = tf.gather(input, axis=1, indices=[0])  # (batch_size, 1, obs_length)
    neighbor_agent_1 = tf.gather(input, axis=1, indices=[1])  # (batch_size, 1, obs_length)
    neighbor_agent_2 = tf.gather(input, axis=1, indices=[2])  # (batch_size, 1, obs_length)
    neighbor_agent_1_friends = tf.gather(input, axis=1, indices=[3, 4])  # (batch_size, 2, obs_length)
    neighbor_agent_2_friends = tf.gather(input, axis=1, indices=[5, 6, 7])  # (batch_size, 3, obs_length)

    #先对neighbor_agent_1进行信息整合
    neighbor_agent_1 = tf.tile(neighbor_agent_1, [1, 2, 1])# (batch_size, 2, obs_length)
    neighbor_agents_1 = tf.concat([neighbor_agent_1, neighbor_agent_1_friends],
                                  axis=-1)# (batch_size, 2, obs_length*2)
    h_ij_1 = Dense(hidden_dim, activation='relu')(neighbor_agents_1)  # batch_size, 2, hidden_dim
    e_ij_1 = Dense(hidden_dim, activation='relu')(neighbor_agents_1)  # batch_size, 2, hidden_dim
    a_ij_1 = tf.nn.softmax(e_ij_1, axis=1)  # batch_size, 2, hidden_dim
    h_i_1 = tf.reduce_sum(a_ij_1 * h_ij_1, axis=1)  # batch_size, hidden_dim

    # 先对neighbor_agent_2进行信息整合
    neighbor_agent_2 = tf.tile(neighbor_agent_2, [1, 3, 1])  # (batch_size, 3, obs_length)
    neighbor_agents_2 = tf.concat([neighbor_agent_2, neighbor_agent_2_friends],
                                  axis=-1)  # (batch_size, 3, obs_length*2)
    h_ij_2 = Dense(hidden_dim, activation='relu')(neighbor_agents_2)  # batch_size, 3, hidden_dim
    e_ij_2 = Dense(hidden_dim, activation='relu')(neighbor_agents_2)  # batch_size, 3, hidden_dim
    a_ij_2 = tf.nn.softmax(e_ij_2, axis=1)  # batch_size, 3, hidden_dim
    h_i_2 = tf.reduce_sum(a_ij_2 * h_ij_2, axis=1)  # batch_size, hidden_dim

    h_i_12 = tf.stack([h_i_1, h_i_2], axis=1)  # batch_size, 2, hidden_dim
    agent = tf.tile(agent, [1, 2, 1])# (batch_size, 2, obs_length)
    h_i_12 = tf.concat([agent, h_i_12], axis=-1)  # (batch_size, 2, obs_length+hidden_dim)
    h_i = Dense(hidden_dim, activation='relu')(h_i_12)  # batch_size, 2, 128
    q_i = Dense(hidden_dim, activation='relu')(h_i_12)  # batch_size, 2, 128
    a_i = tf.nn.softmax(q_i, axis=1)  # batch_size, 2, 128
    h_i_tlt = tf.reduce_sum(a_i * h_i, axis=1)  # batch_size, 128

    xxx = tf.concat([h_i_tlt, tf.reshape(input, [-1, 8*36])], axis=-1)
    concat = Concatenate(axis=-1)([xxx] + actions)
    hidden = Dense(hidden_dim, activation='relu')(concat)
    output = Dense(1)(hidden)
    model = Model(inputs=[input] + actions, outputs=output)

    return model



def Critic_Attention(hidden_dim=128):
    input = Input(shape=(8, 36)) #(batch_size, num_agents, obs_length)
    actions = [Input(shape=(2)) for i in range(8)]

    agent = tf.gather(input, axis=1, indices=[0])#(batch_size, 1, obs_length)
    neighbor_agent_1 = tf.gather(input, axis=1, indices=[1])#(batch_size, 1, obs_length)
    neighbor_agent_2 = tf.gather(input, axis=1, indices=[2])#(batch_size, 1, obs_length)
    neighbor_agent_1_friends = tf.gather(input, axis=1, indices=[3, 4])#(batch_size, 2, obs_length)
    neighbor_agent_2_friends = tf.gather(input, axis=1, indices=[5, 6, 7])#(batch_size, 3, obs_length)

    teammates_obs_embedding = Dense(hidden_dim, activation='relu')
    opponents_obs_embedding = Dense(hidden_dim, activation='relu')
    surround = Dense(hidden_dim, activation='relu')

    #embedding
    # (batch_size, 1, obs_length)
    agent = teammates_obs_embedding(agent)
    # (batch_size, 1, obs_length)
    neighbor_agent_1 = teammates_obs_embedding(neighbor_agent_1)
    # (batch_size, 1, obs_length)
    neighbor_agent_2 = opponents_obs_embedding(neighbor_agent_2)
    # (batch_size, 2, obs_length)
    neighbor_agent_1_friends = teammates_obs_embedding(neighbor_agent_1_friends)
    # (batch_size, 3, obs_length)
    neighbor_agent_2_friends = opponents_obs_embedding(neighbor_agent_2_friends)

    # scale matmul_qk
    dk = tf.cast(hidden_dim, dtype=tf.float32)

    #先对neighbor—agent-1的相关信息进行整合
    # (batch_size, 1, 2)
    matmul_qk_1 = tf.matmul(neighbor_agent_1, neighbor_agent_1_friends, transpose_b=True)
    # (batch_size, 1, 2)
    scaled_attention_logits_1 = matmul_qk_1 / tf.math.sqrt(dk)
    # (batch_size, 1, 2)
    attention_weights_1 = tf.nn.softmax(scaled_attention_logits_1, axis=-1)
    # (batch_size, 1, hidden_dim)
    scaled_attention_1 = tf.matmul(attention_weights_1, neighbor_agent_1_friends)
    # (batch_size, 1, 2*hidden_dim)
    neighbor_agent_1 = tf.concat([neighbor_agent_1, scaled_attention_1], axis=-1)

    # 再对neighbor—agent-2的相关信息进行整合
    # (batch_size, 1, 3)
    matmul_qk_2 = tf.matmul(neighbor_agent_2, neighbor_agent_2_friends, transpose_b=True)
    # (batch_size, 1, 3)
    scaled_attention_logits_2 = matmul_qk_2 / tf.math.sqrt(dk)
    # (batch_size, 1, 3)
    attention_weights_2 = tf.nn.softmax(scaled_attention_logits_2, axis=-1)
    # (batch_size, 1, hidden_dim)
    scaled_attention_2 = tf.matmul(attention_weights_2, neighbor_agent_2_friends)
    # (batch_size, 1, 2*hidden_dim)
    neighbor_agent_2 = tf.concat([neighbor_agent_2, scaled_attention_2], axis=-1)

    # 最后对agent surronding的相关信息进行整合
    # (batch_size, 2, 2*hidden_dim)
    neighbor_agents = tf.concat([neighbor_agent_1, neighbor_agent_2], axis=1)
    # (batch_size, 2, hidden_dim)
    neighbor_agents = surround(neighbor_agents)
    # (batch_size, 1, 2)
    matmul_qk_3 = tf.matmul(agent, neighbor_agents, transpose_b=True)
    # (batch_size, 1, 2)
    scaled_attention_logits_3 = matmul_qk_3 / tf.math.sqrt(dk)
    # (batch_size, 1, 2)
    attention_weights_3 = tf.nn.softmax(scaled_attention_logits_3, axis=-1)
    # (batch_size, 1, hidden_dim)
    scaled_attention_3 = tf.matmul(attention_weights_3, neighbor_agents)
    # (batch_size, 1, 2*hidden_dim)
    agent_surround = tf.concat([agent, scaled_attention_3], axis=-1)

    agent_surround = tf.squeeze(agent_surround, axis=1)
    agent_surround = tf.concat([agent_surround, tf.reshape(input, [-1, 8 * 36])], axis=-1)
    # (batch_size, obs_length+hidden_dim+num_agent*2)
    concat = Concatenate(axis=-1)([agent_surround] + actions)
    concat = Dense(hidden_dim, activation='relu')(concat)
    Q = Dense(1)(concat)

    model = Model(inputs=[input] + actions, outputs=Q)
    return model


def Actor_Attention(hidden_dim=128):
    input = Input(shape=(8, 36))  # (batch_size, num_agents, obs_length)

    agent = tf.gather(input, axis=1, indices=[0])  # (batch_size, 1, obs_length)
    neighbor_agent_1 = tf.gather(input, axis=1, indices=[1])  # (batch_size, 1, obs_length)
    neighbor_agent_2 = tf.gather(input, axis=1, indices=[2])  # (batch_size, 1, obs_length)

    teammates_obs_embedding = Dense(hidden_dim, activation='relu')
    opponents_obs_embedding = Dense(hidden_dim, activation='relu')


    # embedding
    # (batch_size, 1, obs_length)
    agent = teammates_obs_embedding(agent)
    # (batch_size, 1, obs_length)
    neighbor_agent_1 = teammates_obs_embedding(neighbor_agent_1)
    # (batch_size, 1, obs_length)
    neighbor_agent_2 = opponents_obs_embedding(neighbor_agent_2)


    # scale matmul_qk
    dk = tf.cast(hidden_dim, dtype=tf.float32)

    # 最后对agent surronding的相关信息进行整合
    # (batch_size, 2, obs_length)
    neighbor_agents = tf.concat([neighbor_agent_1, neighbor_agent_2], axis=1)
    # (batch_size, 1, 2)
    matmul_qk_3 = tf.matmul(agent, neighbor_agents, transpose_b=True)
    # (batch_size, 1, 2)
    scaled_attention_logits_3 = matmul_qk_3 / tf.math.sqrt(dk)
    # (batch_size, 1, 2)
    attention_weights_3 = tf.nn.softmax(scaled_attention_logits_3, axis=-1)
    # (batch_size, 1, hidden_dim)
    scaled_attention_3 = tf.matmul(attention_weights_3, neighbor_agents)
    # (batch_size, 1, 2*hidden_dim)
    agent_surround = tf.concat([agent, scaled_attention_3], axis=-1)
    # (batch_size, 2*hidden_dim)
    agent_surround = tf.squeeze(agent_surround, axis=1)
    agent_surround = tf.concat([agent_surround, tf.reshape(tf.gather(input, axis=1, indices=[0, 1, 2]), [-1, 3 * 36])], axis=-1)
    agent_surround = Dense(hidden_dim, activation='relu')(agent_surround)
    output = Dense(2, activation='tanh')(agent_surround)  # batch_size, 2
    model = Model(inputs=input, outputs=output)

    return model

def Critic_full_connect(hidden_dim=64):
    input = [Input(shape=(8, 36)) for i in range(8)] #(batch_size, num_agents, obs_length)
    actions = [Input(shape=(2)) for i in range(8)]

    all_agent = [tf.squeeze(tf.gather(agent, axis=1, indices=[0]), axis=1) for agent in input]

    concat = Concatenate(axis=-1)(all_agent + actions)

    hidden = Dense(hidden_dim, activation='relu')(concat)
    hidden = Dense(hidden_dim, activation='relu')(hidden)
    Q = Dense(1)(hidden)

    model = Model(inputs=input + actions, outputs=Q)
    return model


def Actor_full_connect(hidden_dim=64):
    input = Input(shape=(8, 36))  # (batch_size, num_agents, obs_length)

    agent = tf.squeeze(tf.gather(input, axis=1, indices=[0]), axis=1)  # (batch_size, 1, obs_length)

    hidden = Dense(hidden_dim, activation='relu')(agent)
    hidden = Dense(hidden_dim, activation='relu')(hidden)
    output = Dense(2, activation='tanh')(hidden)  # batch_size, 2
    model = Model(inputs=input, outputs=output)

    return model

