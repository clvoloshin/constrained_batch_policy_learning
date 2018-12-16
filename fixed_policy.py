from model import Model
import numpy as np

class FixedPolicy(Model):
    def __init__(self, model, action_space_dim):
        '''
        A fixed manual policy
        '''
        super(FixedPolicy, self).__init__()
        self.model = model
        self.action_space_dim = action_space_dim

    def copy_over_to(self, to_):
        pass

    def predict(self, X_a):
        pass # return [self.model[np.argmax(x_a[:-self.action_space_dim], axis = 1)] == np.argmax(x_a[-self.action_space_dim:], axis=1) for x_a in X_a]

    def fit(self, X, y, verbose=0):
        pass

    def evaluate(self, verbose=False, render=False):
        pass

    def all_actions(self, X):
        return np.array([-np.eye(self.action_space_dim)[self.model[np.argmax(x, axis = 0)]] for x in X])
        