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
from keras_tqdm import TQDMCallback
from model import Model


class NN(Model):
    def __init__(self, num_inputs, num_outputs, dim_of_state, dim_of_actions, gamma, convergence_of_model_epsilon=1e-10):
        '''
        An implementation of fitted Q iteration

        num_inputs: number of inputs
        num_outputs: number of outputs
        dim_of_actions: dimension of action space
        convergence_of_model_epsilon: small float. Defines when the model has converged.
        '''
        super(NN, self).__init__()
        self.convergence_of_model_epsilon = convergence_of_model_epsilon 
        self.model = self.create_model(num_inputs, num_outputs)
        self.dim_of_actions = dim_of_actions
        self.dim_of_state = dim_of_state

        #debug purposes
        self.policy_evalutor = ExactPolicyEvaluator([0], num_inputs-dim_of_actions, gamma)

    def copy_over_to(self, to_):
        # to_.model = keras.models.clone_model(self.model)
        to_.model.set_weights(self.model.get_weights())

    def create_model(self, num_inputs, num_outputs):
        model = Sequential()
        seed = np.random.randint(2**32)
        init = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001, seed=seed)
        model.add(Dense(100, activation='sigmoid', input_shape=(num_inputs,),kernel_initializer=init, bias_initializer=init))
        model.add(Dense(num_outputs, activation='linear',kernel_initializer=init, bias_initializer=init))
        # adam = optimizers.Adam(clipnorm=1.)
        model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])
        return model

    def fit(self, X, y, verbose=0, batch_size=512, epochs=1000, evaluate=True, tqdm_verbose=True, **kw):

        X = self.representation(X[:,0], X[:, 1])
        callbacks_list = [EarlyStoppingByConvergence(epsilon=self.convergence_of_model_epsilon, diff =1e-10, verbose=verbose), TQDMCallback(show_inner=False, show_outer=tqdm_verbose)]
        self.model.fit(X,y,verbose=verbose==2, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, **kw)

        if evaluate:
            return self.evaluate()
        else:
            return None

    def representation(self, *args):
        if len(args) == 1:
            return np.eye(self.dim_of_state)[args[0]]
        elif len(args) == 2:
            return np.hstack([np.eye(self.dim_of_state)[args[0]], np.eye(self.dim_of_actions)[args[1]] ])
        else:
            raise NotImplemented

    def predict(self, X, a):
        return self.model.predict(self.representation(X,a))

    def all_actions(self, X):
        # X_a = ((x_1, a_1)
               # (x_1, a_2)
               #  ....
               # (x_1, a_m)
               # ...
               # (x_N, a_1)
               # (x_N, a_2)
               #  ...
               #  ...
               # (x_N, a_m))
        X = np.array(X)
        X_a = self.cartesian_product(X, np.arange(self.dim_of_actions))


        # Q_x_a = ((Q_x1_a1, Q_x1_a2,... Q_x1_am)
                 # (Q_x2_a1, Q_x2_a2,... Q_x2_am)
                 # ...
                 # (Q_xN_a1, Q_xN_a2,... Q_xN_am)
        # by reshaping using C ordering

        Q_x_a = self.predict(X_a[:,0], X_a[:,1]).reshape(X.shape[0],self.dim_of_actions,order='C')
        return Q_x_a

class EarlyStoppingByConvergence(Callback):
    def __init__(self, monitor='loss', epsilon=0.01, diff=.001, use_both=True, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.epsilon = epsilon
        self.diff = diff
        self.use_both = use_both
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
        
        if self.use_both:
            if ((len(self.losses_so_far) > 1) and (np.abs(self.losses_so_far[-2] - self.losses_so_far[-1]) < self.epsilon)) or (self.losses_so_far[-1] < self.diff):
                self.model.stop_training = True
                self.converged = True
            else:
                pass
        else:
            if ((len(self.losses_so_far) > 1) and (np.abs(self.losses_so_far[-2] - self.losses_so_far[-1]) < self.epsilon)):
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


            
        
