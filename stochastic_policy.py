from model import Model

from keras import backend as K

import numpy as np
from copy import deepcopy

class StochasticPolicy(Model):
    def __init__(self, policy, action_space_dim, policy_evalutor, epsilon=0., prob=None):
        '''
        A fixed manual policy
        '''
        super(StochasticPolicy, self).__init__()
        self.policy = policy

        try:
            has_layer = self.policy.Q.model.get_layer('inp').input
        except:
            has_layer = False

        if has_layer:
            try:
                self.policy.Q.all_actions_func = K.function([self.policy.Q.model.get_layer('inp').input], [self.policy.Q.model.get_layer('dense_2').output])
            except:
                self.policy.Q.all_actions_func = K.function([self.policy.Q.model.get_layer('inp').input], [self.policy.Q.model.get_layer('all_actions').output])

        self.action_space_dim = action_space_dim

        self.epsilon = epsilon
        if prob is not None:
            self.prob = prob
        else:
            self.prob = np.ones(self.action_space_dim)/self.action_space_dim


        #debug purposes
        self.policy_evalutor = policy_evalutor


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

    def all_actions(self, X, x_preprocessed=False,**kw):

        try:
            shape_correct = len(self.policy.Q.model.get_layer('inp').input_shape) == (len(np.array(X).shape))
        except:
            shape_correct = False

        if shape_correct:

            if np.random.random() < self.epsilon:
                arr = -np.eye(self.action_space_dim)[np.random.choice(self.action_space_dim, p=self.prob)]
            else:
                arr = -np.eye(self.action_space_dim)[self.policy.Q([X], x_preprocessed=x_preprocessed)[0]]

            return np.atleast_2d(arr)
        else:
            arr = []
            for x in X:
                if np.random.random() < self.epsilon:
                    arr.append(-np.eye(self.action_space_dim)[np.random.choice(self.action_space_dim, p=self.prob)])
                else:
                    arr.append(-np.eye(self.action_space_dim)[self.policy.Q([x], x_preprocessed=x_preprocessed)[0]])

            return np.atleast_2d(np.array(arr))

