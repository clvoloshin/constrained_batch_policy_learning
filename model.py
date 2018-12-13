"""
Created on December 12, 2018

@author: clvoloshin, 
"""

import keras

class Model(object):
	def __init__(self, num_inputs, num_outputs, dim_of_actions):
		'''
		An implementation of fitted Q iteration

		num_inputs: number of inputs
		num_outputs: number of outputs
		dim_of_actions: dimension of action space
		'''
		self.model = create_model(num_inputs, num_outputs)
		self.dim_of_actions = dim_of_actions

	def create_model(num_inputs, num_outputs):
	    model = Sequential()
	    model.add(Dense(10, activation='relu', input_shape=(num_inputs,)))
	    model.add(Dense(num_outputs, activation='sigmoid'))
	    model.summary()
	    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
	    return model

	def fit(self, X, y):
		self.model.fit(X,y)

	def min_over_a(self, X):
		'''
		Returns min_a Q(X,a), argmin_a Q(X,a)
		'''
		Q_x_a = [self.model.predict(np.hstack([X, np.eye(1, self.dim, action)])) for action in range(self.dim_of_actions)]
		return np.min(Q_x_a), np.argmin(Q_x_a)

	def __call__(self, x, a):
		'''
		Evaluate Q(x,a)
		'''
		return self.model.predict(np.hstack([x, np.eye(1, self.dim, a)]))



