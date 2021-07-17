import numpy as np
import itertools
from make_env import make_env

def obs_process(obs_n):
    '''
    obs_n包含每个智能体的obs，obs是由一个列表组成，列表的长度为2：
    第一维度为智能体的观测值
    第二维度为智能体的relation graph
    # relation graph通过list来表示，list中的值为agent的index，主要分为三层:
        #example: [[0], [2, 7], [[1, 3], [4, 5, 6]]]]
        # 第一层为当前agent的index
        # 第二层为，每个agent类别中，与当前agent最近的agent的index
        # 第三层为，与第二层中agent属于同一类别的agent的index
    '''
    observations_info = []
    relation_graphs = []
    for i, obs in enumerate(obs_n):
        observations_info.append(obs[0])
        relation_graphs.append(obs[1])

    obs_has_relation_info = []
    for relation_graph in relation_graphs:
        obs_i = []
        for index in list(itertools.chain.from_iterable(relation_graph)):
            obs_i.append(observations_info[index])
        obs_has_relation_info.append(np.asarray(obs_i, dtype=np.float32))

    return obs_has_relation_info

if __name__ == '__main__':
    env = make_env(scenario_name="Predator_prey_4v4", discrete_action_space=False, discrete_action_input = False)
    print(env.n)
    print(env.agents)
    print(env.action_space)#[Box(2,), Box(2,), Box(2,)]
    print(env.observation_space)

    obs_n = env.reset()
    obs_n = obs_process(obs_n)
    print(obs_n)
