"""
Created on December 12, 2018

@author: clvoloshin, 
"""
import numpy as np
np.random.seed(3141592)
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from optimization_problem import Program
from value_function import ValueFunction
from fittedq import FittedQIteration
from exponentiated_gradient import ExponentiatedGradient
from fitted_off_policy_evaluation import FittedQEvaluation
from exact_policy_evaluation import ExactPolicyEvaluator
from stochastic_policy import StochasticPolicy
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
map_size = 8
register( id='FrozenLake-no-slip-v0', entry_point='gym.envs.toy_text:FrozenLakeEnv', kwargs={'is_slippery': False, 'map_name':'{0}x{0}'.format(map_size)} )
env = gym.make('FrozenLake-no-slip-v0')
position_of_holes = np.arange(env.desc.shape[0]*env.desc.shape[1]).reshape(env.desc.shape)[np.nonzero(env.desc == 'H')]
position_of_goals = np.arange(env.desc.shape[0]*env.desc.shape[1]).reshape(env.desc.shape)[np.nonzero(env.desc == 'G')]

#### Hyperparam
gamma = 0.9
max_epochs = 1000 # max number of epochs over which to collect data
max_fitting_epochs = 20 #max number of epochs over which to converge to Q^\ast
lambda_bound = 5. # l1 bound on lagrange multipliers
epsilon = .01 # termination condition for two-player game
deviation_from_old_policy_eps = .95 #With what probabaility to deviate from the old policy
# convergence_epsilon = 1e-6 # termination condition for model convergence
action_space_dim = env.nA # action space dimension
state_space_dim = env.nS # state space dimension
eta = 10. # param for exponentiated gradient algorithm
initial_states = [[0]] #The only initial state is [1,0...,0]. In general, this should be a list of initial states
non_terminal_states = np.nonzero(((env.desc == 'S') + (env.desc == 'F')).reshape(-1))[0] # Used for dynamic programming. this is an optimization to make the algorithm run faster. In general, you may not have this
max_number_of_main_algo_iterations = 100 # After how many iterations to cut off the main algorithm
prob = [1/4.]*4 # Probability with which to explore space when deviating from old policy
model_type = 'cnn'

#### Get a decent policy. Called pi_old because this will be the policy we use to gather data
policy_old = None
old_policy_path = os.path.join(model_dir, 'pi_old_map_size_{0}.h5'.format(map_size))
policy_old = DeepQLearning(env, gamma, model_type=model_type,position_of_holes=position_of_holes,position_of_goals=position_of_goals)
if not os.path.isfile(old_policy_path):
    print 'Learning a policy using DQN'
    policy_old.learn()
    policy_old.Q.model.save(old_policy_path)
    print policy_old.Q.evaluate(render=True)
else:
    print 'Loading a policy'
    policy_old.Q.model = load_model(old_policy_path)
    print policy_old.Q.evaluate(render=True)

policy_printer = PrintPolicy(size=[map_size, map_size], env=env)
policy_printer.pprint(policy_old)

#### Problem setup
constraints = [.005, 0]
C = ValueFunction(state_space_dim, non_terminal_states)
G = ValueFunction(state_space_dim, non_terminal_states)
best_response_algorithm = FittedQIteration(state_space_dim + action_space_dim, [map_size, map_size], action_space_dim, max_fitting_epochs, gamma, model_type=model_type)
online_convex_algorithm = ExponentiatedGradient(lambda_bound, len(constraints), eta)
exact_policy_algorithm = ExactPolicyEvaluator(initial_states, state_space_dim, gamma, env)
fitted_off_policy_evaluation_algorithm = FittedQEvaluation(initial_states, state_space_dim + action_space_dim, [map_size, map_size], action_space_dim, max_fitting_epochs, gamma, model_type=model_type)
exploratory_policy_old = StochasticPolicy(policy_old, action_space_dim, exact_policy_algorithm, epsilon=deviation_from_old_policy_eps, prob=prob)
problem = Program(C, G, constraints, action_space_dim, best_response_algorithm, online_convex_algorithm, fitted_off_policy_evaluation_algorithm, exact_policy_algorithm, lambda_bound, epsilon, env, max_number_of_main_algo_iterations,position_of_holes=position_of_holes, position_of_goals=position_of_goals)    
lambdas = []
policies = []

#### Collect Data
num_goal = 0
num_hole = 0
for i in range(max_epochs):
    x = env.reset()
    done = False
    time_steps = 0
    while not done and time_steps < 100:
        time_steps += 1
        
        if exploratory_policy_old is not None:
            action = exploratory_policy_old([x])[0]
            # action = policy_old([x])[0]
            # if np.random.random() < deviation_from_old_policy_eps:
            #     action = np.random.choice(action_space_dim, p = )
        else:
            action = np.random.randint(action_space_dim)
        x_prime , reward, done, _ = env.step(action)

        if done and reward: num_goal += 1
        if done and not reward: num_hole += 1
        c = -reward
        g = [done and not reward, 0]
        problem.collect( x,
                         action,
                         x_prime,
                         c,
                         g,
                         done) #{(x,a,x',c(x,a), g(x,a)^T, done)}


        x = x_prime
    if (i % 200) == 0:
        print 'Epoch: %s' % (i)
problem.finish_collection()

print 'x Distribution:' 
print np.histogram(problem.dataset['x'], bins=np.arange(map_size**2+1)-.5)[0].reshape(map_size,map_size)

print 'x_prime Distribution:' 
print np.histogram(problem.dataset['x_prime'], bins=np.arange(map_size**2+1)-.5)[0].reshape(map_size,map_size)

print 'Number episodes achieved goal: %s. Number episodes fell in hole: %s' % (num_goal, num_hole)
print 'C(pi_old): %s. G(pi_old): %s' % (exact_policy_algorithm.run(exploratory_policy_old,policy_is_greedy=False) )

number_of_total_state_action_pairs = (state_space_dim-np.sum(env.desc=='H')-np.sum(env.desc=='G'))*action_space_dim
number_of_state_action_pairs_seen = len(np.unique(problem.dataset['state_action'],axis=0))
print 'Percentage of State/Action space seen: %s' % (number_of_state_action_pairs_seen/float(number_of_total_state_action_pairs))

### Solve Batch Constrained Problem
iteration = 0
while not problem.is_over(policies, lambdas):
    iteration += 1
    policy_printer.pprint(policies)
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

problem.save()


