"""
Created on December 12, 2018

@author: clvoloshin, 
"""

import numpy as np

class Program(object):
	def __init__(self, C, G, constraints, dim, best_response_algorithm, online_convex_algorithm, fitted_off_policy_evaluation_algorithm, lambda_bound = 1., epsilon = .01):
		'''
		This is a problem of the form: min_pi C(pi) where G(pi) < eta.

		dataset: list. Will be {(x,a,x',c(x,a), g(x,a)^T)}
		dim: number of constraints
		C, G: dictionary of |A| dim vectors
		best_response_algorithm: function which accepts a |A| dim vector and outputs a policy which minimizes L
		online_convex_algorithm: function which accepts a policy and returns an |A| dim vector (lambda) which maximizes L
		lambda_bound: positive int. l1 bound on lambda |lambda|_1 <= B
		constraints:  |A| dim vector
		epsilon: small positive float. Denotes when this problem has been solved.
		'''

		self.dataset = []
		self.constraints = constraints
		self.C = C
		self.G = G
		self.dim = len(constraints)
		self.lambda_bound = lambda_bound
		self.epsilon = epsilon
		self.best_response_algorithm = best_response_algorithm
		self.online_convex_algorithm = online_convex_algorithm
		self.fitted_off_policy_evaluation_algorithm = fitted_off_policy_evaluation_algorithm

	def best_response(self, lamb):
		'''
		Best-response(lambda) = argmin_{pi} L(pi, lambda) 
		'''
		dataset = [[(x,a,x_prime, c + np.dot(lamb, g-self.constraints))] for (x,a,x_prime,c,g) in self.dataset]
		policy = self.best_response_algorithm.run(dataset)
		return policy

	def online_algo(self, policy):
		'''
		No regret online convex optimization routine
		'''

		gradient = np.vstack([self.G.last() - self.constraints, np.array([0])])
		lambda_t = self.online_convex_algorithm.run(gradient)
		return lambda_t

	def lagrangian(self, lamb):
		# C(pi) + lambda^T (G(pi) - eta), where eta = constraints, pi = avg of all pi's seen
		return self.C.avg() + np.dot(lamb, (self.G.avg() - self.constraints))

	def max_of_lagrangian_over_lambda():
		'''
		The maximum of C(pi) + lambda^T (G(pi) - eta) over lambda is
		B*e_{k+1}, all the weight on the phantom index if G(pi) < eta for all constraints
		B*e_k otherwise where B is the l1 bound on lambda and e_k is the standard
		basis vector putting full mass on the constraint which is violated the most
		'''

		maximum = np.max(self.G.avg() - self.constraints)
		index = np.argmax(self.G.avg() - self.constraints) 

		if maximum > 0:
			lamb = self.lambda_bound * np.eye(1, self.dim, index)
		else:
			lamb = np.zeros(1, self.dim)
			lamb[-1] = self.lambda_bound

		return self.lagrangian(lamb)

	def min_of_lagrangian_over_policy(lamb):
		'''
		This function evaluates L(best_response(avg_lambda), avg_lambda)
		'''

		br = self.best_response(lamb)
		return self.C(br) + np.dot(lamb, (self.G(br) - self.constraints))

	def update(self, policy):
		
		#update C

		C_pi = self.fitted_off_policy_evaluation_algorithm.run(self.dataset[:,:-1], policy)
		self.C.append(C_pi)

		idxs = np.ones((len(self.dataset),), bool)
		all_except_C = idxs
		all_except_C[-2] = False
		G_pi = self.fitted_off_policy_evaluation_algorithm.run(self.dataset[:,all_except_C], policy)
		self.G.append(G_pi)



	def is_over(self, lambdas):
		# lambdas: list. We care about average of all lambdas seen thus far
		# If |max_lambda L(avg_pi, lambda) - L(best_response(avg_lambda), avg_lambda)| < epsilon, then done
		if len(lambdas) == 0: return False

		x = self.max_of_lagrangian_over_lambda()
		y = self.min_of_lagrangian_over_policy(np.mean(lambdas, 0))

		if np.abs(x-y) < self.epsilon:
			return True
		else:
			return False





