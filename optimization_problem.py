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
		dataset = [[x,a,x_prime, c + np.dot(lamb, g-self.constraints)] for (x,a,x_prime,c,g) in self.dataset]
		policy = self.best_response_algorithm.run(np.array(dataset))
		return policy

	def online_algo(self):
		'''
		No regret online convex optimization routine
		'''
		gradient = self.G.last() - self.constraints
		lambda_t = self.online_convex_algorithm.run(gradient)
		return lambda_t

	def lagrangian(self, lamb):
		# C(pi) + lambda^T (G(pi) - eta), where eta = constraints, pi = avg of all pi's seen
		return self.C.avg() + np.dot(lamb, (self.G.avg() - self.constraints))

	def max_of_lagrangian_over_lambda(self):
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
			lamb = np.zeros(self.dim)
			lamb[-1] = self.lambda_bound

		return self.lagrangian(lamb)

	def min_of_lagrangian_over_policy(self, lamb):
		'''
		This function evaluates L(best_response(avg_lambda), avg_lambda)
		'''
		
		print 'Calculating best-response(lambda_avg)'
		best_policy = self.best_response(lamb)

		print 'Calculating C(best_response(lambda_avg))'
		C_br = self.fitted_off_policy_evaluation_algorithm.run(self.dataset[:,:-1], best_policy)
		
		print 'Calculating G(best_response(lambda_avg))'
		G_br = []
		for i in range(self.dim-1):
			g = np.vstack(self.dataset[:,-1])[:,i].reshape(-1,1)
			G_br.append(self.fitted_off_policy_evaluation_algorithm.run(np.hstack([self.dataset[:,:-2], g]), best_policy))
		G_br.append(0)
		G_br = np.array(G_br)
		
		return C_br + np.dot(lamb, (G_br - self.constraints))

	def update(self, policy):
		
		#update C
		C_pi = self.fitted_off_policy_evaluation_algorithm.run(self.dataset[:,:-1], policy)
		self.C.append(C_pi)

		#update G
		G_pis = []
		for i in range(self.dim-1):
			g = np.vstack(self.dataset[:,-1])[:,i].reshape(-1,1)
			G_pis.append(self.fitted_off_policy_evaluation_algorithm.run(np.hstack([self.dataset[:,:-2], g]), policy))
		G_pis.append(0)

		self.G.append(G_pis)

	def collect(self, data):
		'''
		Add more data
		'''
		self.dataset.append(data)

	def finish_collection(self):
		# preprocess
		self.dataset = np.array(self.dataset)


	def is_over(self, lambdas):
		# lambdas: list. We care about average of all lambdas seen thus far
		# If |max_lambda L(avg_pi, lambda) - L(best_response(avg_lambda), avg_lambda)| < epsilon, then done
		if len(lambdas) == 0: return False

		
		x = self.max_of_lagrangian_over_lambda()
		y = self.min_of_lagrangian_over_policy(np.mean(lambdas, 0))

		difference = np.abs(x-y)
		print 'max L: %s, min_L: %s, difference: %s' % (x,y,np.abs(x-y))
		if difference < self.epsilon:
			return True
		else:
			return False





