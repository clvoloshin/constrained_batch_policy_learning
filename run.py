"""
Created on December 12, 2018

@author: clvoloshin, 
"""
import numpy as np
np.random.seed(3141592)
import tensorflow as tf
from optimization_problem import Program
from value_function import ValueFunction
from fittedq import FittedQIteration
from exponentiated_gradient import ExponentiatedGradient
from fitted_off_policy_evaluation import FittedQEvaluation
from exact_policy_evaluation import ExactPolicyEvaluator
from optimal_policy import DeepQLearning
from print_policy import PrintPolicy
from keras.models import load_model

###
#paths
import os
model_dir = os.path.join(os.getcwd(), 'models')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
###

#### Setup Gym 
import gym
from gym.envs.registration import register
register( id='FrozenLake-no-slip-v0', entry_point='gym.envs.toy_text:FrozenLakeEnv', kwargs={'is_slippery': False, 'map_name':'4x4'} )
env = gym.make('FrozenLake-no-slip-v0')

#### Hyperparam
gamma = 0.9
max_epochs = 20 # max number of epochs over which to collect data
max_fitting_epochs = 10 #max number of epochs over which to converge to Q^\ast
lambda_bound = 10. # l1 bound on lagrange multipliers
epsilon = .01 # termination condition for two-player game
deviation_from_old_policy_eps = .4 #With what probabaility to deviate from the old policy
# convergence_epsilon = 1e-6 # termination condition for model convergence
action_space_dim = env.nA # action space dimension
state_space_dim = env.nS # state space dimension
eta = 10. # param for exponentiated gradient algorithm
initial_states = [np.eye(1, state_space_dim, 0)] #The only initial state is [1,0...,0]. In general, this should be a list of initial states

#### Get a decent policy. Called pi_old because this will be the policy we use to gather data
policy_old = None
old_policy_path = os.path.join(model_dir, 'pi_old.h5')
policy_old = DeepQLearning(env, gamma)
if not os.path.isfile(old_policy_path):
    print 'Learning a policy using DQN'
    policy_old.learn()
    policy_old.Q.model.save(old_policy_path)
    print policy_old.Q.evaluate(render=True)
else:
    print 'Loading a policy'
    policy_old.Q.model = load_model(old_policy_path)
    print policy_old.Q.evaluate(render=True)
PrintPolicy().pprint(policy_old.Q)

#### Problem setup
constraints = [.01, 0]
C = ValueFunction()
G = ValueFunction()
best_response_algorithm = FittedQIteration(state_space_dim + action_space_dim, action_space_dim, max_fitting_epochs, gamma)
online_convex_algorithm = ExponentiatedGradient(lambda_bound, len(constraints), eta)
exact_policy_algorithm = ExactPolicyEvaluator(initial_states, state_space_dim, env, gamma)
fitted_off_policy_evaluation_algorithm = FittedQEvaluation(initial_states, state_space_dim + action_space_dim, action_space_dim, max_fitting_epochs, gamma)
problem = Program(C, G, constraints, action_space_dim, best_response_algorithm, online_convex_algorithm, fitted_off_policy_evaluation_algorithm, exact_policy_algorithm, lambda_bound, epsilon, env)    
lambdas = []
policies = []

#### Collect Data
num_goal = 0
num_hole = 0
for i in range(max_epochs):
    x = env.reset()
    done = False
    time_steps = 0
    while not done:
        time_steps += 1
        
        if policy_old is not None:
            action = policy_old.Q(np.eye(1, state_space_dim, x))[0]
            if np.random.random() < deviation_from_old_policy_eps:
                action = np.random.randint(action_space_dim)
        else:
            action = np.random.randint(action_space_dim)
        x_prime , reward, done, _ = env.step(action)

        if done and reward: num_goal += 1
        if done and not reward: num_hole += 1
        c = -reward
        g = [done and not reward, 0]
        problem.collect( np.eye(1, state_space_dim, x).reshape(-1).tolist(),
                         action,
                         np.eye(1, state_space_dim, x_prime).reshape(-1).tolist(),
                         c,
                         g,
                         done) #{(x,a,x',c(x,a), g(x,a)^T, done)}


        x = x_prime
    print 'Epoch: %s. Num steps: %s. Avg episode length: %s' % (i, time_steps, float(len(problem.dataset)/(i+1)))
print 'Number episodes achieved goal: %s. Number episodes fell in hole: %s' % (num_goal, num_hole)
problem.finish_collection()

### Solve Batch Constrained Problem
iteration = 0
while not problem.is_over(lambdas):
    iteration += 1
    print '*'*20
    print 'Iteration %s' % iteration
    print
    if len(lambdas) == 0:
        # first iteration
        lambdas.append(online_convex_algorithm.get())
        print 'lambda_{0} = {1}'.format(iteration, lambdas[-1])
    else:
        # all other iterations
        lambda_t = problem.online_algo()
        lambdas.append(lambda_t)
        print 'lambda_{0} = online-algo(pi_{1}) = {2}'.format(iteration, iteration-1, lambdas[-1])

    lambda_t = lambdas[-1]
    pi_t = problem.best_response(lambda_t, desc='FQI pi_{0}'.format(iteration))

    policies.append(pi_t)
    problem.update(pi_t, iteration) #Evaluate C(pi_t), G(pi_t) and save

    
