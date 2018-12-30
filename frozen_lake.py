import gym
import numpy as np
from gym.envs.registration import register
from gym.envs.toy_text import FrozenLakeEnv


class ExtendedFrozenLake(FrozenLakeEnv):
    def __init__(self, early_termination, desc=None, map_name="4x4",is_slippery=True):
        super(ExtendedFrozenLake, self).__init__(desc=desc, map_name=map_name, is_slippery=is_slippery)
        self.deterministic = True
        self.max_time_steps = early_termination
        self.min_cost = -1. #set by env
        self.env_type = 'lake'

    def is_early_episode_termination(self, cost=None, time_steps=None, total_cost=None):
        if time_steps > self.max_time_steps:
            return True, 0.
        else:
            return False, 0.

    def step(self, a):
        transitions = self.P[self.s][a]
        i = self.categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d= transitions[i]
        self.s = s
        self.lastaction=a

        c = -r
        g = [int(d and not r)]
        return (s, (c,g), d, {"prob" : p})

    @staticmethod
    def categorical_sample(prob_n, np_random):
        """
        Sample from categorical distribution
        Each row specifies class probabilities
        """
        prob_n = np.asarray(prob_n)
        csprob_n = np.cumsum(prob_n)
        return (csprob_n > np_random.rand()).argmax()
