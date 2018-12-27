"""
Created on December 12, 2018

@author: clvoloshin, 
"""
from pyvirtualdisplay import Display
import numpy as np
np.random.seed(3141592)
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from optimization_problem import Program
from fittedq import *
from exponentiated_gradient import ExponentiatedGradient
from fitted_off_policy_evaluation import *
from exact_policy_evaluation import ExactPolicyEvaluator
from stochastic_policy import StochasticPolicy
from DQN import DeepQLearning
from print_policy import PrintPolicy
from keras.models import load_model
from keras import backend as K
from env_dqns import *

def main(env_name, headless):

    if headless:
        display = Display(visible=0, size=(1280, 1024))
        display.start()
    ###
    #paths
    import os
    model_dir = os.path.join(os.getcwd(), 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    ###

    if env_name == 'lake':
        from config_lake import *
    elif env_name == 'car':
        from config_car import *
    else:
        raise

    #### Get a decent policy. 
    #### Called pi_old because this will be the policy we use to gather data
    policy_old = None
    old_policy_path = os.path.join(model_dir, old_policy_name)
    
    if env_name == 'lake':
        policy_old = LakeDQN(env, 
                             gamma, 
                             action_space_map = action_space_map, 
                             model_type=model_type,
                             position_of_holes=position_of_holes,
                             position_of_goals=position_of_goals, 
                             max_time_spent_in_episode=max_time_spent_in_episode,
                             num_iterations = num_iterations,
                             sample_every_N_transitions = sample_every_N_transitions,
                             batchsize = batchsize,
                             copy_over_target_every_M_training_iterations = copy_over_target_every_M_training_iterations,
                             buffer_size = buffer_size,
                             num_frame_stack=num_frame_stack,
                             min_buffer_size_to_train=min_buffer_size_to_train,
                             )
    elif env_name == 'car':
        policy_old = CarDQN(env, 
                            gamma, 
                            action_space_map = action_space_map, 
                            action_space_dim=action_space_dim, 
                            model_type=model_type,
                            max_time_spent_in_episode=max_time_spent_in_episode,
                            num_iterations = num_iterations,
                            sample_every_N_transitions = sample_every_N_transitions,
                            batchsize = batchsize,
                            copy_over_target_every_M_training_iterations = copy_over_target_every_M_training_iterations,
                            buffer_size = buffer_size,
                            min_epsilon = min_epsilon,
                            initial_epsilon = initial_epsilon,
                            epsilon_decay_steps = epsilon_decay_steps,
                            num_frame_stack=num_frame_stack,
                            min_buffer_size_to_train=min_buffer_size_to_train,
                            )



    else:
        raise
    
    if not os.path.isfile(old_policy_path):
        print 'Learning a policy using DQN'
        policy_old.learn()
        policy_old.Q.model.save(old_policy_path)
        print policy_old.Q.evaluate(render=True, environment_is_dynamic=stochastic_env)
    else:
        print 'Loading a policy'
        policy_old.Q.model = load_model(old_policy_path)
        print policy_old.Q.evaluate(render=True, environment_is_dynamic=stochastic_env)

    if env_name == 'lake':
        policy_printer = PrintPolicy(size=[map_size, map_size], env=env)
        policy_printer.pprint(policy_old)

    #### Problem setup
    constraints = [.1, 0]
    if env_name == 'lake':
        best_response_algorithm = LakeFittedQIteration(state_space_dim + action_space_dim, 
                                                       [map_size, map_size], 
                                                       action_space_dim, 
                                                       max_Q_fitting_epochs, 
                                                       gamma, 
                                                       model_type=model_type, 
                                                       position_of_goals=position_of_goals, 
                                                       position_of_holes=position_of_holes,
                                                       num_frame_stack=num_frame_stack)
        
        fitted_off_policy_evaluation_algorithm = LakeFittedQEvaluation(initial_states, 
                                                           state_space_dim + action_space_dim, 
                                                           [map_size, map_size], 
                                                           action_space_dim, 
                                                           max_eval_fitting_epochs, 
                                                           gamma, 
                                                           model_type=model_type, 
                                                           position_of_goals=position_of_goals, 
                                                           position_of_holes=position_of_holes,
                                                           num_frame_stack=num_frame_stack)
    elif env_name == 'car':
        best_response_algorithm = CarFittedQIteration(state_space_dim, 
                                                      action_space_dim, 
                                                      max_Q_fitting_epochs, 
                                                      gamma, 
                                                      model_type=model_type,
                                                      num_frame_stack=num_frame_stack)
        fitted_off_policy_evaluation_algorithm = CarFittedQEvaluation(state_space_dim, 
                                                                      action_space_dim, 
                                                                      max_eval_fitting_epochs, 
                                                                      gamma, 
                                                                      model_type=model_type,
                                                                      num_frame_stack=num_frame_stack)
    else:
        raise

    online_convex_algorithm = ExponentiatedGradient(lambda_bound, len(constraints), eta)
    exact_policy_algorithm = ExactPolicyEvaluator(action_space_map, gamma, env=env, num_frame_stack=num_frame_stack)
    exploratory_policy_old = StochasticPolicy(policy_old, 
                                              action_space_dim, 
                                              exact_policy_algorithm, 
                                              epsilon=deviation_from_old_policy_eps, 
                                              prob=prob)
    problem = Program(constraints, 
                      action_space_dim, 
                      best_response_algorithm, 
                      online_convex_algorithm, 
                      fitted_off_policy_evaluation_algorithm, 
                      exact_policy_algorithm, 
                      lambda_bound, 
                      epsilon, 
                      env, 
                      max_number_of_main_algo_iterations,
                      num_frame_stack)    

    lambdas = []
    policies = []

    #### Collect Data
    num_goal = 0
    num_hole = 0
    for i in range(max_epochs):
        x = env.reset()
        problem.collect(x, start=True)

        done = False
        time_steps = 0
        while not done:
            time_steps += 1
            

            action = exploratory_policy_old(problem.dataset.current_state())[0]
            x_prime, cost, done, _ = env.step(action_space_map[action])

            done = done or env.is_early_episode_termination(time_steps)
            # if done and reward: num_goal += 1
            # if done and not reward: num_hole += 1
            c = cost[0]
            g = cost[1] + [0]
            problem.collect( action,
                             x_prime,
                             np.hstack([c,g]).reshape(-1).tolist(),
                             done
                             ) #{(x,a,x',c(x,a), g(x,a)^T, done)}


            x = x_prime
        if (i % 200) == 0:
            print 'Epoch: %s' % (i)
    problem.finish_collection(env_name)

    # print 'x Distribution:' 
    # print np.histogram(problem.dataset['x'], bins=np.arange(map_size**2+1)-.5)[0].reshape(map_size,map_size)

    # print 'x_prime Distribution:' 
    # print np.histogram(problem.dataset['x_prime'], bins=np.arange(map_size**2+1)-.5)[0].reshape(map_size,map_size)

    # print 'Number episodes achieved goal: %s. Number episodes fell in hole: %s' % (num_goal, num_hole)
    # print 'C(pi_old): %s. G(pi_old): %s' % (exact_policy_algorithm.run(exploratory_policy_old,policy_is_greedy=False) )

    # number_of_total_state_action_pairs = (state_space_dim-np.sum(env.desc=='H')-np.sum(env.desc=='G'))*action_space_dim
    # number_of_state_action_pairs_seen = len(np.unique(problem.dataset['state_action'],axis=0))
    # print 'Percentage of State/Action space seen: %s' % (number_of_state_action_pairs_seen/float(number_of_total_state_action_pairs))

    ### Solve Batch Constrained Problem
    iteration = 0
    while not problem.is_over(policies, lambdas, infinite_loop=True):
        iteration += 1
        K.clear_session()
        # policy_printer.pprint(policies)
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

        # policies.append(pi_t)
        problem.update(pi_t, iteration) #Evaluate C(pi_t), G(pi_t) and save

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='Choose environment.')
    parser.add_argument('-env', dest='env', help='lake/car openAI environment')
    parser.add_argument('--headless', dest='headless', action='store_true',
                        help = 'Use flag if running on server so you can run render() from openai')
    parser.set_defaults(headless=False)
    args = parser.parse_args()
    

    assert args.env in ['lake', 'car'], 'Need to choose between FrozenLakeEnv (lake) or Car Racing (car) environment'


    main(args.env, args.headless)