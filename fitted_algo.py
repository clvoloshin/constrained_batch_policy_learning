"""
Created on December 12, 2018

@author: clvoloshin, 
"""

from neural_network import NN
import numpy as np

class FittedAlgo(object):
    def __init__(self, num_inputs, dim_of_states, dim_of_actions, max_epochs, gamma):
        '''
        An implementation of fitted Q iteration

        num_inputs: number of inputs
        dim_of_actions: dimension of action space
        max_epochs: positive int, specifies how many iterations to run the algorithm
        gamma: discount factor
        '''
        self.num_inputs = num_inputs
        self.dim_of_states = dim_of_states
        self.dim_of_actions = dim_of_actions
        self.max_epochs = max_epochs
        self.gamma = gamma

    def init_Q(self, epsilon=1e-10):
        return NN(self.num_inputs, 1, self.dim_of_states, self.dim_of_actions, self.gamma, epsilon)

    def fit(self, X, y, epsilon=1e-10, **kw):
        # D_k = {(X,y)} is the dataset of the kth iteration of Fitted Q
        # self.Q_k = self.init_Q(epsilon)
        self.Q_k.epsilon = epsilon
        self.Q_k.fit(X, y, **kw)

    def skim(self, X_a, x_prime):
        full_set = np.hstack([X_a, x_prime.reshape(1,-1).T])
        idxs = np.unique(full_set, axis=0, return_index=True)[1]
        return idxs

    def run(self, dataset):
        '''
        Abstract function
        '''
        pass


