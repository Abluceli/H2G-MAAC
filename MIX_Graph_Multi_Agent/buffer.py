from collections import deque
import random
class ReplayBuffer():
    def __init__(self, max_len):
        self.buffer = deque(maxlen=max_len)
        self.length = 0

    def reset(self):
        self.buffer.clear()
        self.length = 0

    def experience(self, obs_n, act_n, rew_n, obs_next_n, done_n):
        self.buffer.append([obs_n, act_n, rew_n, obs_next_n, done_n])
        self.length += 1

    def sample(self, batch_size):
        data = random.sample(self.buffer, batch_size)