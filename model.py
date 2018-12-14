"""
Created on December 12, 2018

@author: clvoloshin, 
"""

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import mean_squared_error
from keras import optimizers
from keras.callbacks import Callback, TensorBoard
from exact_policy_evaluation import ExactPolicyEvaluator
import itertools


import gym
from gym.envs.registration import register
register( id='FrozenLake-no-slip-v1', entry_point='gym.envs.toy_text:FrozenLakeEnv', kwargs={'is_slippery': False, 'map_name':'4x4'} )
env = gym.make('FrozenLake-no-slip-v1')

class Model(object):
    def __init__(self, num_inputs, num_outputs, dim_of_actions, gamma, convergence_of_model_epsilon=1e-5):
        '''
        An implementation of fitted Q iteration

        num_inputs: number of inputs
        num_outputs: number of outputs
        dim_of_actions: dimension of action space
        convergence_of_model_epsilon: small float. Defines when the model has converged.
        '''
        self.convergence_of_model_epsilon = convergence_of_model_epsilon 
        self.model = self.create_model(num_inputs, num_outputs)
        self.dim_of_actions = dim_of_actions

        #debug purposes
        self.policy_evalutor = ExactPolicyEvaluator([np.eye(1, num_inputs-dim_of_actions, 0)], num_inputs-dim_of_actions, env, gamma)

    def create_model(self, num_inputs, num_outputs):
        model = Sequential()
        init = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001, seed=1)
        model.add(Dense(5, activation='relu', input_shape=(num_inputs,),kernel_initializer=init, bias_initializer=init))
        model.add(Dense(num_outputs, activation='linear',kernel_initializer=init, bias_initializer=init))
        # adam = optimizers.Adam(clipnorm=1.)
        model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])
        return model

    def fit(self, X, y, epochs=None, verbose=False):

        callbacks_list = [EarlyStoppingByConvergence(epsilon=self.convergence_of_model_epsilon, verbose=verbose)]
        if epochs is None:
            self.model.fit(X,y,verbose=verbose, epochs=1000, callbacks=callbacks_list)
        else:
            self.model.fit(X,y,verbose=verbose,epochs=epochs,callbacks=callbacks_list)

        return self.evaluate()

    def evaluate(self, verbose=False, render=False):
        return self.policy_evalutor.run(self, verbose=verbose, render=render)

    def min_over_a(self, X, randomized_tiebreaking=False):
        '''
        Returns min_a Q(X,a), argmin_a Q(X,a)
        '''

        Q_x_a = self.all_actions(X)
        return self.min_and_argmin(Q_x_a, randomized_tiebreaking, axis=1)

    def max_over_a(self, X, randomized_tiebreaking=False):
        '''
        Returns min_a Q(X,a), argmin_a Q(X,a)
        '''

        Q_x_a = self.all_actions(X)
        return self.max_and_argmax(Q_x_a, randomized_tiebreaking, axis=1)

    def all_actions(self, X):
        # X_a = ((x_1, a_1)
               # (x_2, a_1)
               #  ....
               # (x_N, a_1)
               # (x_1, a_2)
               #  ...
               # (x_N, a_2)
               #  ...
               # (x_N, a_m))
        X_a = self.cartesian_product(X, np.eye(self.dim_of_actions))


        # Q_x_a = ((Q_x1_a1, Q_x1_a2,... Q_x1_am)
                 # (Q_x2_a1, Q_x2_a2,... Q_x2_am)
                 # ...
                 # (Q_xN_a1, Q_xN_a2,... Q_xN_am)
        # by reshaping using fortran ordering
        Q_x_a = self.model.predict(X_a).reshape(X.shape[0],self.dim_of_actions,order='F')
        return Q_x_a

    @staticmethod
    def max_and_argmax(Q, randomized_tiebreaking=False, **kw):
        ''' max + Argmax + Breaks max/argmax ties randomly'''
        if not randomized_tiebreaking:
            return np.max(Q, **kw), np.argmax(Q, **kw)
        else:
            tie_breaker = np.random.random(Q.shape) * (Q==Q.max())
            argmax = np.argmax(tie_breaker, **kw) # this is counter intuitive.
            return Q[argmax], argmax

    @staticmethod
    def min_and_argmin(Q, randomized_tiebreaking=False, **kw):
        ''' min + Argmin + Breaks min/argmin ties randomly'''
        if not randomized_tiebreaking:
            return np.min(Q, **kw), np.argmin(Q, **kw)
        else:
            tie_breaker = - np.random.random(Q.shape) * (Q==Q.min())
            argmin = np.argmin(tie_breaker, **kw)
            return Q[argmin], argmin

    def __call__(self, *args):
        if len(args) == 1:
            '''
            Run policy: pi = argmin_a Q(x,a)
            '''
            x = args[0]
            return self.min_over_a(x, False)[1]
        elif len(args) == 2:
            '''
            Evaluate Q(x,a)
            '''
            x,a = args
            return self.model.predict(np.hstack([x, np.eye(self.dim_of_actions)[a]  ]))
        else:
            # Not implemented.
            raise

    @staticmethod
    def cartesian_product(x,y):
        return np.hstack([np.tile(x.T, y.shape[1]).T, np.tile(y,x.shape[0]).reshape(-1,y.shape[1])])

class EarlyStoppingByConvergence(Callback):
    def __init__(self, monitor='loss', epsilon=0.01, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.epsilon = epsilon
        self.verbose = verbose
        self.losses_so_far = []
        self.converged = False

    def on_epoch_end(self, epoch, logs={}):
        self.epoch = epoch
        
        current = logs.get(self.monitor)
        if current is None:
            print("Early stopping requires %s available!" % self.monitor)
            exit()
        else:
            self.losses_so_far.append(current)

        if self.verbose:
            if (self.epoch % 100) == 0:
                print 'Epoch %s, loss: %s' % (epoch, self.losses_so_far[-1])
        if (len(self.losses_so_far) > 1) and (np.abs(self.losses_so_far[-2] - self.losses_so_far[-1]) < self.epsilon):
            self.model.stop_training = True
            self.converged = True
        else:
            pass

    def on_train_end(self, logs=None):
        if self.epoch > 1:
            if self.verbose > 0:
                if self.converged:
                    print 'Epoch %s: early stopping. Converged. Delta: %s. Loss: %s' % (self.epoch, np.abs(self.losses_so_far[-2] - self.losses_so_far[-1]), self.losses_so_far[-1])
                else:
                    print 'Epoch %s. NOT converged. Delta: %s. Loss: %s' % (self.epoch, np.abs(self.losses_so_far[-2] - self.losses_so_far[-1]), self.losses_so_far[-1])

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.losses_so_far = []
        self.converged = False


            
        
