"""
Created on December 15, 2018

@author: clvoloshin, 
"""
import numpy as np
np.set_printoptions(suppress=True)
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
map_size = [4,4]
register( id='FrozenLake-no-slip-v0', entry_point='gym.envs.toy_text:FrozenLakeEnv', kwargs={'is_slippery': False, 'map_name':'{0}x{1}'.format(map_size[0], map_size[1])} )
env = gym.make('FrozenLake-no-slip-v0')
position_of_holes = np.arange(env.desc.shape[0]*env.desc.shape[1]).reshape(env.desc.shape)[np.nonzero(env.desc == 'H')]
position_of_goals = np.arange(env.desc.shape[0]*env.desc.shape[1]).reshape(env.desc.shape)[np.nonzero(env.desc == 'G')]

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
initial_states = [[0]] #The only initial state is [1,0...,0]. In general, this should be a list of initial states
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
PrintPolicy(env=env).pprint(policy_old)

# model_dict = {0: 1, 4: 1, 8: 0}
# for i in range(grid_size*grid_size):
#     if i not in model_dict:
#         model_dict[i] = np.random.randint(action_space_dim)
# policy_old = FixedPolicy(model_dict, action_space_dim, policy_evaluator)
# PrintPolicy().pprint(policy_old)

### Policy to evaluate
model_dict = {0: 1, 4: 1, 8: 2, 9: 1, 13: 2, 14: 2}
for i in range(map_size[0]*map_size[1]):
    if i not in model_dict:
        model_dict[i] = np.random.randint(action_space_dim)
policy = FixedPolicy(model_dict, action_space_dim, policy_evaluator)

print 'Evaluate this policy:'
PrintPolicy(env=env).pprint(policy)

#### Problem setup

def main(policy_old, policy, model_type='cnn'):

    fqi = FittedQIteration(state_space_dim + action_space_dim, map_size, action_space_dim, max_fitting_epochs, gamma,model_type =model_type )
    fqe = FittedQEvaluation(initial_states, state_space_dim + action_space_dim, map_size, action_space_dim, max_fitting_epochs, gamma,model_type =model_type )
    ips = InversePropensityScorer(action_space_dim)
    exact_evaluation = ExactPolicyEvaluator(initial_states, state_space_dim, gamma, env)

    max_epochs = np.array([1000]) # np.arange(50,1060,100) # max number of epochs over which to collect data
    epsilons = np.array([.25]) # np.array([.5])
    trials = np.array([1,2]) # np.arange(20) 
    eps_epochs_trials = cartesian_product(epsilons, max_epochs,trials)
    
    all_trials_estimators = []
    for epsilon in epsilons:

        trials_estimators = []
        for epochs in max_epochs:

            trial_estimators = []
            for trial in trials: 
                estimators = run_trial(policy_old, policy, epochs, epsilon, fqi, fqe, ips, exact_evaluation)
                
                trial_estimators.append(estimators)
            trials_estimators.append(trial_estimators)

        all_trials_estimators.append(trials_estimators)

        # print epsilon, np.mean(all_trials_evaluated[-1]), np.mean(all_trials_approx_ips[-1]), np.mean(all_trials_exact_ips[-1]), np.mean(all_trials_exact[-1])
    
    results = np.hstack([eps_epochs_trials, np.array(all_trials_estimators).reshape(-1, np.array(all_trials_estimators).shape[-1])])
    df = pd.DataFrame(results, columns=['epsilon', 'num_trajectories', 'trial_num', 'exact','fqe'])
    df.to_csv('fqe_quality.csv', index=False)

def run_trial(policy_old, policy, epochs, epsilon, fqi, fqe, ips, exact_evaluation):
    #### Collect Data
    num_goal = 0
    num_hole = 0
    dataset = Dataset([0], action_space_dim)
    dataset_removed = Dataset([0], action_space_dim)
    
    data = []
    mapping = {0:np.array([0,-1]), 2:np.array([0,1]), 1:np.array([1,0]), 3:np.array([-1,0])}
    for x in set(np.nonzero(env.desc.reshape(-1) == 'F')[0]).union(set(np.nonzero(env.desc.reshape(-1) == 'S')[0])) :
        for action in range(4):

            # if x == 4: import pdb; pdb.set_trace()
            row = int(x/map_size[1])
            col = int(x - row*int(map_size[1]))

            new_row, new_col  = np.array([row, col]) + mapping[action]
            if (new_row < 0) or (new_row > (map_size[0]-1)):
                new_row, new_col = row, col
            elif (new_col < 0) or (new_col > (map_size[1]-1)):
                new_row, new_col = row, col
            else:
                pass
            x_prime = new_row*map_size[1] + new_col

            if (env.desc[new_row, new_col] == 'H') or (env.desc[new_row, new_col] == 'G'): 
                done = True
            else:
                done = False

            if env.desc[new_row, new_col] == 'G': 
                goal = True
            else:
                goal = False

            data.append([x,action,x_prime,-goal,done and not goal,done])

    
    for idx, datum in enumerate(data):
        count = idx % 4

        if count == 0:
            must_keep = 0
            kept = 0

        
        if (count == 3) and (kept == 0):
            must_keep = 1

        if (not must_keep) and (np.random.choice([0,1], p=[epsilon, 1-epsilon])):
            kept += 1
            dataset.append(*datum)
        else:
            dataset_removed.append(*datum)

        
    dataset.preprocess()
    dataset_removed.preprocess()
    
    print 'Distribution:' 
    print np.histogram(dataset['x'], bins=np.arange(map_size[0]*map_size[1]+1)-.5)[0].reshape(map_size)

    print 'Distribution:' 
    print np.histogram(dataset['x_prime'], bins=np.arange(map_size[0]*map_size[1]+1)-.5)[0].reshape(map_size)
    

    dataset.set_cost('c')
    dataset_removed.set_cost('c')
    
    # Exact
    exact = exact_evaluation.run(policy)[0]
    print exact

    # Importance Sampling
    # approx_ips, exact_ips, approx_pdis, exact_pdis = ips.run(dataset, policy, policy_old, epsilon, gamma)
    
    # FQE

    for eps in [1e-3]:

        evaluated = []
        for i in range(1):
            evaluated.append(fqe.run(dataset, policy, epochs=5000, epsilon=eps, desc='FQE epsilon %s' % np.round(epsilon,2),position_of_holes=position_of_holes, position_of_goals=position_of_goals))
            PrintPolicy(env=env).pprint(fqe.Q_k)

            print evaluated[-1]

        evaluated = np.mean(evaluated)
        print evaluated

        print np.mean((fqe.Q_k(dataset['x'], dataset['a']).T - (dataset['cost'] + gamma*fqe.Q_k(dataset['x_prime'], policy(dataset['x_prime']) )[0]*(1-dataset['done'])))**2)
        print np.vstack([dataset['x'], dataset['a'], np.round((fqe.Q_k(dataset['x'], dataset['a']).T - (dataset['cost'] + gamma*fqe.Q_k(dataset['x_prime'], policy(dataset['x_prime']) )[0]*(1-dataset['done'])))**2, 2)]).T
        if len(dataset_removed['x']) > 0:
            print np.mean((fqe.Q_k(dataset_removed['x'], dataset_removed['a']).T - (dataset_removed['cost'] + gamma*fqe.Q_k(dataset_removed['x_prime'], policy(dataset_removed['x_prime']))[0]*(1-dataset_removed['done'])))**2)

    df = pd.DataFrame(np.vstack([dataset['x'], dataset['a'], dataset['x_prime'], dataset['cost'], dataset['done'], np.round(fqe.Q_k(dataset['x'], dataset['a']),3).T, np.around(dataset['cost'] + gamma*fqe.Q_k(dataset['x_prime'], policy(dataset['x_prime'])).T*(1-dataset['done']),2)  , (fqe.Q_k(dataset['x'], dataset['a']).T - (dataset['cost'] + gamma*fqe.Q_k(dataset['x_prime'], policy(dataset['x_prime'])).T*(1-dataset['done']) ))  ]).T, columns = ['x','a','x_prime','c','done','Q(x,a)', 'Q(x_,pi(x_))', 'diff'])
    df_outside = pd.DataFrame(np.vstack([dataset_removed['x'], dataset_removed['a'], dataset_removed['x_prime'], dataset_removed['cost'], dataset_removed['done'], np.round(fqe.Q_k(dataset_removed['x'], dataset_removed['a']),3).T, np.around(dataset_removed['cost'] + gamma*fqe.Q_k(dataset_removed['x_prime'], policy(dataset_removed['x_prime'])).T*(1-dataset_removed['done']),2)  , (fqe.Q_k(dataset_removed['x'], dataset_removed['a']).T - (dataset_removed['cost'] + gamma*fqe.Q_k(dataset_removed['x_prime'], policy(dataset_removed['x_prime'])).T*(1-dataset_removed['done']) ))  ]).T, columns = ['x','a','x_prime','c','done','Q(x,a)', 'Q(x_,pi(x_))', 'diff'])
    print exact, evaluated

    return exact-exact, evaluated-exact

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def create_df(array, **kw):
    return pd.DataFrame(array, **kw)


def custom_plot(x, y, minimum, maximum, **kwargs):
    ax = kwargs.pop('ax', plt.gca())
    base, = ax.plot(x, y, **kwargs)
    ax.fill_between(x, minimum, maximum, facecolor=base.get_color(), alpha=0.15)

main(policy_old, policy)
df = pd.read_csv('fqe_quality.csv')
for epsilon, group in df.groupby('epsilon'):
    del group['epsilon']
    # group.set_index('num_trajectories').plot()
    # import pdb; pdb.set_trace()
    means = group.groupby('num_trajectories').mean()
    stds = group.groupby('num_trajectories').std()


    del means['trial_num']
    del stds['trial_num']

    print '*'*20
    print 'Epsilon: %s' % epsilon
    print means
    print stds

    fig, ax = plt.subplots(1)
    colors = ['red', 'green', 'blue']
    for i, col in enumerate(['fqe']):
        # import pdb; pdb.set_trace()

        x = np.array(means.index)
        mu = np.array(means[col])
        sigma = np.array(stds[col])

        lower_bound = mu + sigma
        upper_bound = mu - sigma

        custom_plot(x, mu, lower_bound, upper_bound, marker='o', label=col, color=colors[i])
        


    # means.plot(yerr=stds)

    # plt.title(epsilon)
    ax.legend()
    ax.set_title('Probability of exploration: %s' % epsilon)
    ax.set_xlabel('Number of trajectories in dataset')
    ax.set_ylabel('Policy Evaluation Error')
    plt.show()