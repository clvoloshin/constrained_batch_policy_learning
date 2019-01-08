"""
Created on December 12, 2018

@author: clvoloshin, 
"""

import numpy as np

class ValueFunction(object):
    def __init__(self):
        '''
        '''
        self.prev_values = []
        self.exact_values = []
        self.eval_values = {}
        # self.V = {}
        # self.dim_state_space = dim_state_space
        # self.non_terminal_states = non_terminal_states

    def append(self, *args):
        if len(args) == 1:
            value = args[0]
            self.prev_values.append(value)
        elif len(args) == 2:
            value, policy = args
            self.prev_values.append(value)
            # self.V[self.vectorize(policy)] = value

    def avg(self, append_zero=False):
        if append_zero:
            return np.hstack([np.mean(self.prev_values, 0), np.array([0])])
        else:
            return np.mean(self.prev_values, 0)

    def last(self, append_zero=False):
        if append_zero:
            return np.hstack([self.prev_values[-1], np.array([0])])
        else:
            return np.array(self.prev_values[-1])

    def add_exact_values(self, values):
        self.exact_values.append(values)

    def add_eval_values(self, eval_values, idx):
        if idx not in self.eval_values:
            self.eval_values[idx] = []
        
        self.eval_values[idx].append(eval_values)


    # def vectorize(self, policy):
    #     # Can be done for low dim discrete spaces
    #     return tuple(policy(self.non_terminal_states))

    # def __getitem__(self, policy):
    #     pi = self.vectorize(policy)
    #     if pi in self.V:
    #         return np.array(self.V[pi])
    #     else:
    #         raise KeyError

    # def __contains__(self, policy):
    #     pi = self.vectorize(policy)
    #     if pi in self.V:
    #         return True
    #     else:
    #         return False
