from model import Model


import numpy as np
from copy import deepcopy

class FixedPolicy(Model):
    def __init__(self, policy, action_space_dim, policy_evalutor):
        '''
        A fixed manual policy
        '''
        super(FixedPolicy, self).__init__()
        self.policy = policy
        self.action_space_dim = action_space_dim

        #debug purposes
        self.policy_evalutor = deepcopy(policy_evalutor)
        self.Q = None
        self.get_Q_val()

    def get_Q_val(self):
        self.policy_evalutor.initial_states = np.hstack([np.nonzero((self.policy_evalutor.env.desc == 'S').reshape(-1))[0], np.nonzero((self.policy_evalutor.env.desc == 'F').reshape(-1))[0]])
        self.Q_tmp = self.policy_evalutor.get_Qs(self)

        self.Q = {}
        for idx, state in enumerate(self.policy_evalutor.initial_states):
            self.Q[state] = np.eye(self.action_space_dim)[self.policy[state]]*(self.Q_tmp[idx]-1e-7)

    def copy_over_to(self, to_):
        pass

    def predict(self, X_a):
        pass # return [self.model[np.argmax(x_a[:-self.action_space_dim], axis = 1)] == np.argmax(x_a[-self.action_space_dim:], axis=1) for x_a in X_a]

    def fit(self, X, y, verbose=0):
        pass

    def representation(self, *args):
        if len(args) == 1:
            return args[0]
        elif len(args) == 2:
            return args[0], args[1]
        else:
            raise NotImplemented

    def all_actions(self, X):
        if self.Q is None:
            return np.array([-np.eye(self.action_space_dim)[self.policy[x]] for x in X])
        else:
            arr = []
            for x in X:
                try:
                    arr.append(self.Q[x])
                except:
                    arr.append([0]*self.action_space_dim)
            return np.array(arr)

