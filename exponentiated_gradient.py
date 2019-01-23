

import numpy as np

class ExponentiatedGradient(object):
    def __init__(self, lambda_bound, number_of_constraints, eta=1., starting_lambda='uniform'):
        '''
        '''
        self.eta = eta
        self.lambda_bound = lambda_bound
        self.number_of_constraints = number_of_constraints
        if starting_lambda == 'uniform':
            self.w_t = self.lambda_bound*np.ones(self.number_of_constraints)/self.number_of_constraints
        else:
            self.w_t = starting_lambda
            self.lambda_bound = np.sum(starting_lambda)
    
    def run(self, gradient):
        self.w_t = self.w_t/self.lambda_bound
        unnormalized_wt = self.w_t*np.exp(self.eta*gradient) # positive since working  w/ costs.
        self.w_t = self.lambda_bound*unnormalized_wt/sum(unnormalized_wt)
        return self.w_t
    
    def get(self):
        return self.w_t