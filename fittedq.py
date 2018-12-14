"""
Created on December 12, 2018

@author: clvoloshin, 
"""

from fitted_algo import FittedAlgo
from model import Model
import numpy as np

from print_policy import PrintPolicy

class FittedQIteration(FittedAlgo):
	def __init__(self, num_inputs, dim_of_actions, max_epochs, gamma):
		'''
		An implementation of fitted Q iteration

		num_inputs: number of inputs
		dim_of_actions: dimension of action space
		max_epochs: positive int, specifies how many iterations to run the algorithm
		gamma: discount factor
		'''
		super(FittedQIteration, self).__init__(num_inputs, dim_of_actions, max_epochs, gamma)


	def run(self, dataset):
		# dataset is the original dataset generated by pi_{old} to which we will find
		# an approximately optimal Q

		self.Q_k = self.init_Q()
		for k in range(self.max_epochs):
			
			# {((x,a), c+gamma*min_a Q(x',a))}
			costs = dataset['cost'] + self.gamma*self.Q_k.min_over_a(dataset['x_prime'])[0]
			X_a = dataset['state_action']
			
			
			PrintPolicy().pprint(X_a, costs)
			PrintPolicy().pprint(self.Q_k)

			#Calc c-lambda(g-nu) evaluation
			import pdb;pdb.set_trace()
			# idxs = np.unique(X_a, axis=0, return_index=True)[1]
			# x_a_c = np.hstack([np.argmax(X_a[idxs][:,:-4],1).reshape(1,-1).T, np.argmax(X_a[idxs][:,-4:],1).reshape(1,-1).T, costs[idxs].reshape(1,-1).T])
			
			self.fit(X_a, costs)
			# import pdb;pdb.set_trace()
			# tmp = np.hstack([x_a_c, self.Q_k.model.predict(X_a[idxs])])
			# print np.mean((self.Q_k.model.predict(X_a).T[0] - costs)**2)
			# PrintPolicy().pprint(dataset['state_action'], dataset['cost'] + self.gamma*self.Q_k.min_over_a(dataset['x_prime'])[0])

		return self.Q_k

