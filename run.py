"""
Created on December 12, 2018

@author: clvoloshin, 
"""

import gym
import numpy as np
import tensorflow as tf
from optimization_problem import Program
from value_function import ValueFunction
from fittedq import FittedQIteration
from exponentiated_gradient import ExponentiatedGradient
from fitted_off_policy_evaluation import FittedOffPolicyEvaluation

from gym.envs.registration import register
register( id='FrozenLake-no-slip-v0', entry_point='gym.envs.toy_text:FrozenLakeEnv', kwargs={'is_slippery': False, 'map_name':'8x8'} )
env = gym.make('FrozenLake-no-slip-v0')
import pdb; pdb.set_trace()


#### Hyperparam
gamma = 0.9
episodes = 100 # max number of episodes
max_epochs = 1000 # max number of epochs
lambda_bound = 1. # l1 bound on lagrange multipliers
epsilon = .01 # termination condition for two-player game
action_space_dim = env.nA # action space dimension
state_space_dim = env.nS # state space dimension
eta = .5 # param for exponentiated gradient algorithm
initial_states = [np.eye(1, state_space_dim, 0)] #The only initial state is [1,0...,0]. In general, this should be a list of initial states

#### Problem setup
constraints = [.01, 0]
C = ValueFunction()
G = ValueFunction()
best_response_algorithm = FittedQIteration(state_space_dim + action_space_dim, action_space_dim, max_epochs, gamma)
online_convex_algorithm = ExponentiatedGradient(lambda_bound, len(constraints), eta)
fitted_off_policy_evaluation_algorithm = FittedOffPolicyQEvaluation(initial_states, state_space_dim + action_space_dim, action_space_dim, max_epochs, gamma)
problem = Program(C, G, constraints, action_space_dim, best_response_algorithm, online_convex_algorithm, fitted_off_policy_evaluation_algorithm, lambda_bound, epsilon)    
lambdas = []
policies = []


#### Collect Data

for _ in range(max_epochs):
    
    x = env.reset()
    done = False
    reward = 0
    while not done:
        action = np.random.randint(action_space_dim)        
        x_prime , reward, done, _ = env.step(action)
        problem.collect([np.eye(1, state_space_dim, x),
                         action,
                         np.eye(1, state_space_dim, x_prime),
                         -reward,
                         np.array([done, 0]) ]) #{(x,a,x',c(x,a), g(x,a)^T)}


### Solve Batch Constrained Problem
while not problem.is_over(lambdas):

    if len(lambdas) == 0:
        # first iteration
        lambdas.append(online_convex_algorithm.get())
    else:
        # all other iterations
        lambda_t = problem.online_algo(policies)
        lambdas.append(lambda_t)

    lambda_t = lambdas[-1]
    pi_t = problem.best_response(lambda_t)
    policies.append(pi_t)

    problem.update(pi_t) #Evaluate C(pi_t), G(pi_t) and save

    
