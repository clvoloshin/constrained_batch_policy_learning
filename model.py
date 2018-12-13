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


class Model(object):
	def __init__(self, num_inputs, num_outputs, dim_of_actions):
		'''
		An implementation of fitted Q iteration

		num_inputs: number of inputs
		num_outputs: number of outputs
		dim_of_actions: dimension of action space
		'''
		self.model = self.create_model(num_inputs, num_outputs)
		self.dim_of_actions = dim_of_actions

	def create_model(self, num_inputs, num_outputs):
	    model = Sequential()
	    model.add(Dense(5, activation='relu', input_shape=(num_inputs,)))
	    model.add(Dense(num_outputs, activation='relu'))
	    # model.summary()
	    model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])
	    return model

	def fit(self, X, y):
		self.model.fit(X,y,verbose=0)

	def min_over_a(self, X):
		'''
		Returns min_a Q(X,a), argmin_a Q(X,a)
		'''
		Q_x_a = [self.model.predict(np.hstack([X, np.eye(1, self.dim_of_actions, action)])) for action in range(self.dim_of_actions)]
		return np.min(Q_x_a), np.argmin(Q_x_a)

	def __call__(self, *args):
		if len(args) == 1:
			'''
			Run policy: pi = argmin_a Q(x,a)
			'''
			x = args[0]
			return self.min_over_a(x)[1]
		elif len(args) == 2:
			'''
			Evaluate Q(x,a)
			'''
			x,a = args
			return self.model.predict(np.hstack([x, np.eye(1, self.dim_of_actions, a)]))
		else:
			# Not implemented.
			raise

