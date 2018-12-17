"""
Created on December 15, 2018

@author: clvoloshin, 
"""
import numpy as np
np.random.seed(314)
import tensorflow as tf
from optimization_problem import Dataset
from fittedq import FittedQIteration
from fixed_policy import FixedPolicy
from fitted_off_policy_evaluation import FittedQEvaluation
from exact_policy_evaluation import ExactPolicyEvaluator
from inverse_propensity_scoring import InversePropensityScorer
from exact_policy_evaluation import ExactPolicyEvaluator
from optimal_policy import DeepQLearning
from print_policy import PrintPolicy
from keras.models import load_model
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


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
grid_size = 4
register( id='FrozenLake-no-slip-v0', entry_point='gym.envs.toy_text:FrozenLakeEnv', kwargs={'is_slippery': False, 'map_name':'{0}x{0}'.format(grid_size)} )
env = gym.make('FrozenLake-no-slip-v0')

#### Hyperparam
gamma = 0.9
max_fitting_epochs = 10 #max number of epochs over which to converge to Q^\ast
lambda_bound = 10. # l1 bound on lagrange multipliers
epsilon = .01 # termination condition for two-player game
deviation_from_old_policy_eps = .7 #With what probabaility to deviate from the old policy
# convergence_epsilon = 1e-6 # termination condition for model convergence
action_space_dim = env.nA # action space dimension
state_space_dim = env.nS # state space dimension
eta = 10. # param for exponentiated gradient algorithm
initial_states = [np.eye(1, state_space_dim, 0)] #The only initial state is [1,0...,0]. In general, this should be a list of initial states
policy_evaluator = ExactPolicyEvaluator(initial_states, state_space_dim, gamma)

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

print 'Old Policy'
PrintPolicy().pprint(policy_old.Q)

### Policy to evaluate
model_dict = {0: 1, 4: 1, 8: 2, 9: 1, 13: 2, 14: 2}
for i in range(grid_size*grid_size):
    if i not in model_dict:
        model_dict[i] = np.random.randint(action_space_dim)
policy = FixedPolicy(model_dict, action_space_dim, policy_evaluator)

print 'Evaluate this policy:'
PrintPolicy().pprint(policy)

#### Problem setup

def main(policy_old, policy):
    fqi = FittedQIteration(state_space_dim + action_space_dim, action_space_dim, max_fitting_epochs, gamma)
    fqe = FittedQEvaluation(initial_states, state_space_dim + action_space_dim, action_space_dim, max_fitting_epochs, gamma)
    ips = InversePropensityScorer(action_space_dim)
    exact_evaluation = ExactPolicyEvaluator(initial_states, state_space_dim, gamma, env)

    max_epochs = np.array([10, 100, 1000]) # max number of epochs over which to collect data
    epsilons = np.array([.01, .1, .5, .9])
    trials = np.arange(10)
    eps_epochs_trials = cartesian_product(epsilons, max_epochs,trials)
    
    
    all_trials_exact = []
    all_trials_evaluated = []
    all_trials_exact_ips = []
    all_trials_approx_ips = []
    
    for epsilon in epsilons:

        trials_exact = []
        trials_evaluated = []
        trials_exact_ips = []
        trials_approx_ips = []
        for epochs in max_epochs:

            trial_exact = []
            trial_evaluated = []
            trial_exact_ips = []
            trial_approx_ips = []
            for trial in trials: 
                exact, evaluated, exact_ips, approx_ips = run_trial(policy_old, policy, epochs, epsilon, fqi, fqe, ips, exact_evaluation)
                trial_exact.append(exact)
                trial_evaluated.append(evaluated)
                trial_exact_ips.append(exact_ips)
                trial_approx_ips.append(approx_ips)
            trials_exact.append(trial_exact)
            trials_evaluated.append(trial_evaluated)
            trials_exact_ips.append(trial_exact_ips)
            trials_approx_ips.append(trial_approx_ips)

        all_trials_exact.append(trials_exact)
        all_trials_evaluated.append(trials_evaluated)
        all_trials_exact_ips.append(trials_exact_ips)
        all_trials_approx_ips.append(trials_approx_ips)
        

        print epsilon, np.mean(all_trials_evaluated[-1]), np.mean(all_trials_approx_ips[-1]), np.mean(all_trials_exact_ips[-1]), np.mean(all_trials_exact[-1])

    df = create_df(eps_epochs_trials[:,0].tolist(), 
                   eps_epochs_trials[:,1].tolist(),
                   eps_epochs_trials[:,2].tolist(), 
                   all_trials_exact, 
                   all_trials_evaluated, 
                   all_trials_exact_ips, 
                   all_trials_approx_ips, 
                   columns=['epsilon', 'num_trajectories', 'trial_num', 'exact','fqe','exact_ips', 'approx_ips'])
    df.to_csv('fqe_quality.csv', index=False)

def run_trial(policy_old, policy, epochs, epsilon, fqi, fqe, ips, exact_evaluation):
    #### Collect Data
    num_goal = 0
    num_hole = 0
    dataset = Dataset([0], action_space_dim)
    for i in range(epochs):
        x = env.reset()
        done = False
        time_steps = 0
        while not done:
            time_steps += 1
            
            if policy_old is not None:
                action = policy_old.Q(np.eye(1, state_space_dim, x))[0]
                if np.random.random() < epsilon:
                    action = np.random.randint(action_space_dim)
            else:
                action = np.random.randint(action_space_dim)
            x_prime , reward, done, _ = env.step(action)

            if done and reward: num_goal += 1
            if done and not reward: num_hole += 1
            c = -reward
            g = [done and not reward, 0]
            dataset.append( np.eye(1, state_space_dim, x).reshape(-1).tolist(),
                             action,
                             np.eye(1, state_space_dim, x_prime).reshape(-1).tolist(),
                             c,
                             g,
                             done) #{(x,a,x',c(x,a), g(x,a)^T, done)}


            x = x_prime
        # print 'Epoch: %s. Num steps: %s. Avg episode length: %s' % (i, time_steps, float(len(problem.dataset)/(i+1)))
    dataset.preprocess()
    print 'Epsilon %s. Number goals: %s. Number holes: %s.' % (epsilon, num_goal, num_hole)
    print 'Distribution:' 
    print np.histogram(np.argmax(dataset['x_prime'],1), bins=np.arange(grid_size**2+1)-.5)[0].reshape(grid_size,grid_size)
    

    dataset.set_cost('c')
    
    # Exact
    exact = exact_evaluation.run(policy)[0]

    # Importance Sampling
    approx_ips, exact_ips = ips.run(dataset, policy, policy_old.Q, epsilon, gamma)
    
    # FQE
    evaluated = fqe.run(dataset, policy, epochs=5000, epsilon=1e-13, desc='FQE epsilon %s' % np.round(epsilon,2) )
    # evaluated = 0

    return exact-exact, evaluated-exact, approx_ips-exact, exact_ips-exact

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def create_df(*arrays, **kw):
    return pd.DataFrame(np.hstack([np.vstack(x).reshape(1,-1).T for x in arrays]), **kw)

main(policy_old, policy)
df = pd.read_csv('fqe_quality.csv')
for epsilon, group in df.groupby('epsilon'):
    del group['epsilon']
    # group.set_index('num_trajectories').plot()
    import pdb; pdb.set_trace()
    means = group.groupby('num_trajectories').mean()
    stds = group.groupby('num_trajectories').std()
    del means['trial_num']
    del stds['trial_num']

    means.plot(yerr=stds)

    plt.title(epsilon)
    plt.show()
