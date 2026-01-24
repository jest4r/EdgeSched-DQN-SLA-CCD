import numpy as np
import gym
from gym import spaces

class TaskSchedulingEnv(gym.Env):
    def __init__(self, num_tasks=10, num_servers=3):
        super(TaskSchedulingEnv, self).__init__()
        self.num_tasks = num_tasks
        self.num_servers = num_servers
        self.action_space = spaces.Discrete(num_servers)
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_tasks, num_servers), dtype=np.float32)
        self.state = np.random.rand(num_tasks, num_servers)

    def reset(self):
        self.state = np.random.rand(self.num_tasks, self.num_servers)
        return self.state

    def step(self, action):
        reward = np.random.rand()
        done = True
        info = {}
        next_state = np.random.rand(self.num_tasks, self.num_servers)
        return next_state, reward, done, info

    def render(self, mode='human'):
        pass
