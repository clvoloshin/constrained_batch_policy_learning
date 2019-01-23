
from keras import backend as K
import numpy as np

class FittedAlgo(object):
    def __init__(self):
        '''
        An implementation of fitted Q iteration

        num_inputs: number of inputs
        dim_of_actions: dimension of action space
        max_epochs: positive int, specifies how many iterations to run the algorithm
        gamma: discount factor
        '''

    def init_Q(self):
        '''
        Absract function
        '''
        pass

    def fit(self, X, y, epsilon=1e-10, **kw):
        # D_k = {(X,y)} is the dataset of the kth iteration of Fitted Q
        # self.Q_k = self.init_Q(epsilon)
        # K.set_value(self.Q_k.model.optimizer.iterations, 0)
        self.Q_k.epsilon = epsilon
        self.Q_k.fit(X, y, **kw)

    def fit_generator(self, generator, epsilon=1e-10, **kw):
        # D_k = {(X,y)} is the dataset of the kth iteration of Fitted Q
        # self.Q_k = self.init_Q(epsilon)
        # K.set_value(self.Q_k.model.optimizer.iterations, 0)
        self.Q_k.epsilon = epsilon
        self.Q_k.fit_generator(generator, **kw)

    def skim(self, X_a, x_prime):
        full_set = np.hstack([X_a, x_prime.reshape(1,-1).T])
        idxs = np.unique(full_set, axis=0, return_index=True)[1]
        return idxs

    def run(self, dataset):
        '''
        Abstract function
        '''
        pass


