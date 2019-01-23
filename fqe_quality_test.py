
import numpy as np
np.random.seed(0)
import tensorflow as tf
from optimization_problem import Dataset
from fittedq import LakeFittedQIteration as FittedQIteration
from fixed_policy import FixedPolicy
from fitted_off_policy_evaluation import LakeFittedQEvaluation as FittedQEvaluation
from exact_policy_evaluation import ExactPolicyEvaluator
from inverse_propensity_scoring import InversePropensityScorer
from env_dqns import LakeDQN
from print_policy import PrintPolicy
from keras.models import load_model
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras import backend as K

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
map_size = [8]
# register( id='FrozenLake-no-slip-v0', entry_point='gym.envs.toy_text:FrozenLakeEnv', kwargs={'is_slippery': False, 'map_name':'{0}x{1}'.format(map_size[0], map_size[1])} )
# env = gym.make('FrozenLake-no-slip-v0')
from frozen_lake import ExtendedFrozenLake
env = ExtendedFrozenLake(100, map_name = '{0}x{0}'.format(map_size[0]), is_slippery= False)
position_of_holes = np.arange(env.desc.shape[0]*env.desc.shape[1]).reshape(env.desc.shape)[np.nonzero(env.desc == 'H')]
position_of_goals = np.arange(env.desc.shape[0]*env.desc.shape[1]).reshape(env.desc.shape)[np.nonzero(env.desc == 'G')]

#### Hyperparam
gamma = 0.9
max_fitting_epochs = 100 #max number of epochs over which to converge to Q^\ast
lambda_bound = 10. # l1 bound on lagrange multipliers
action_space_dim = env.nA # action space dimension
state_space_dim = env.nS # state space dimension
eta = 10. # param for exponentiated gradient algorithm
initial_states = [[0]] #The only initial state is [1,0...,0]. In general, this should be a list of initial states
from config_lake import action_space_map, frame_skip, num_frame_stack, pic_size

policy_evaluator = ExactPolicyEvaluator(action_space_map, gamma, env=env, frame_skip=frame_skip, num_frame_stack=num_frame_stack, pic_size = pic_size)
dqn_model_type = 'mlp'
testing_model_type = 'mlp'

#### Get a decent policy. Called pi_old because this will be the policy we use to gather data
policy_old = None
old_policy_path = os.path.join(model_dir, 'pi_old_map_size_{0}_{1}.h5'.format(map_size[0], dqn_model_type))
policy_old = LakeDQN(env, gamma, model_type=dqn_model_type,position_of_holes=position_of_holes,position_of_goals=position_of_goals, min_epsilon=0, initial_epsilon=1, epsilon_decay_steps=100)
if not os.path.isfile(old_policy_path):
    print 'Learning a policy using DQN'
    policy_old.learn()
    policy_old.Q.model.save(old_policy_path)
    # print policy_old.Q.evaluate(render=True)
else:
    print 'Loading a policy'
    policy_old.Q.model = load_model(old_policy_path)
    # print policy_old.Q.evaluate(render=True)

print 'Old Policy'
map_size = [map_size[0]]*2
policy_printer = PrintPolicy(size=map_size, env=env)
policy_printer.pprint(policy_old)

# model_dict = {0: 1, 4: 1, 8: 0}
# for i in range(grid_size*grid_size):
#     if i not in model_dict:
#         model_dict[i] = np.random.randint(action_space_dim)
# policy_old = FixedPolicy(model_dict, action_space_dim, policy_evaluator)
# PrintPolicy().pprint(policy_old)

### Policy to evaluate
# model_dict = {0: 1, 4: 1, 8: 2, 9: 1, 13: 2, 14: 2} # 4x4
# model_dict = {0: 1, 8: 1, 16: 1, 24: 1, 32: 1, 40: 1, 48: 1, 56: 2, 57: 2, 
                # 58: 3, 50: 2, 51: 3, 43: 2, 44: 2, 45: 1, 53: 2}#,  61: 2, 62: 2} # 8x8
# model_dict = {0: 2, 1: 2, 2: 1, 10: 1, 18: 1, 26:1, 34:1}
model_dict = {0: 1, 8: 1, 16: 1, 24: 1, 32: 1, 40: 1, 48: 1, 56: 2, 57: 3, 
                58: 3, 50: 2, 51: 3, 43: 2, 44: 2, 45: 1, 53: 2}#,  61: 2, 62: 2} # 8x8
# model_dict = {0: 1, 8: 1, 16: 2, 17: 2, 18: 2} # 8x8
# model_dict = {0: 1, 8: 1, 16: 1, 24: 1, 32: 1, 40: 1, 48: 2} # 8x8
for i in range(map_size[0]*map_size[1]):
    if i not in model_dict:
        model_dict[i] = np.random.randint(action_space_dim)
policy = FixedPolicy(model_dict, action_space_dim, policy_evaluator)

print 'Evaluate this policy:'
policy_printer.pprint(policy)
#### Problem setup

def main(policy_old, policy, model_type='mlp'):

    fqi = None #FittedQIteration(state_space_dim + action_space_dim, map_size, action_space_dim, max_fitting_epochs, gamma,model_type =model_type )
    fqe = FittedQEvaluation(initial_states, state_space_dim + action_space_dim, map_size, action_space_dim, max_fitting_epochs, gamma,model_type =model_type )
    ips = InversePropensityScorer(env, state_space_dim, action_space_dim, map_size)
    exact_evaluation = ExactPolicyEvaluator(action_space_map, gamma, env=env, frame_skip=frame_skip, num_frame_stack=num_frame_stack, pic_size = pic_size)

    max_percentage = np.arange(.1, 1.05, .1) # max number of epochs over which to collect data
    epsilons = np.array([.95])
    trials = np.arange(128) 
    eps_epochs_trials = cartesian_product(epsilons, max_percentage,trials)
    
    all_trials_estimators = []
    for epsilon in epsilons:

        trials_estimators = []
        dataset, exact = get_dataset(1500, epsilon, exact_evaluation)
            
        for trial_num, trial in enumerate(trials): 

            trial_estimators = []
            for perc_num, percentage in enumerate(max_percentage):
                K.clear_session()
                idxs = np.random.permutation(np.arange(len(dataset.episodes))).tolist()
                estimators = run_trial(idxs, dataset, policy_old, policy, percentage, epsilon, fqi, fqe, ips, exact)
                
                trial_estimators.append(estimators)
            trials_estimators.append(trial_estimators)
            results = np.hstack([cartesian_product(epsilons, max_percentage,trials[0:(trial_num+1)]), np.array([trials_estimators]).reshape(-1, np.array([trials_estimators]).shape[-1])])
            df = pd.DataFrame(results, columns=['epsilon', 'num_trajectories', 'trial_num', 'exact','fqe','approx_ips', 'exact_ips','approx_pdis', 'exact_pdis', 'doubly_robust', 'weighted_doubly_robust', 'AM'])
            df.to_csv('fqe_quality_fixed_dr.csv', index=False)

        all_trials_estimators.append(trials_estimators)

        # print epsilon, np.mean(all_trials_evaluated[-1]), np.mean(all_trials_approx_ips[-1]), np.mean(all_trials_exact_ips[-1]), np.mean(all_trials_exact[-1])
    
    results = np.hstack([eps_epochs_trials, np.array(all_trials_estimators).reshape(-1, np.array(all_trials_estimators).shape[-1])])
    df = pd.DataFrame(results, columns=['epsilon', 'num_trajectories', 'trial_num', 'exact','fqe','approx_ips', 'exact_ips','approx_pdis', 'exact_pdis', 'doubly_robust', 'weighted_doubly_robust', 'AM'])
    df.to_csv('fqe_quality_fixed_dr.csv', index=False)

def run_trial(idxs, dataset, policy_old, policy, percentage, epsilon, fqi, fqe, ips, exact):
    #### Collect Data
    policy_old.Q.model = load_model(old_policy_path)

    
    done = False
    maximum = len(np.unique(np.hstack([dataset['x'].reshape(1,-1).T,  dataset['a'].reshape(1,-1).T, dataset['x_prime'].reshape(1,-1).T ]), axis=0))
    all_episodes = dataset.episodes
    sampled_episodes = []
    count = 0
    if (percentage >= 1.):
        sampled_episodes = all_episodes
        num_unique = len(np.unique(np.hstack([np.hstack([x['x'] for x in sampled_episodes]).reshape(1,-1).T, np.hstack([x['a'] for x in sampled_episodes]).reshape(1,-1).T]), axis=0))
    else:
        while (not done) and (len(idxs) >= count):
            count += 1
            idx = idxs[count]
            sampled_episodes.append(all_episodes[idx])

            num_unique = len(np.unique(np.hstack([np.hstack([x['x'] for x in sampled_episodes]).reshape(1,-1).T, np.hstack([x['a'] for x in sampled_episodes]).reshape(1,-1).T]), axis=0))
            if (float(num_unique)/maximum) > percentage:
                episode= sampled_episodes.pop()
                idx = 0
                while 1:
                    idx += 1
                    new_episode = {k:val[:idx] for k,val in episode.iteritems()}
                    sampled_episodes.append(new_episode)
                    num_unique = len(np.unique(np.hstack([np.hstack([x['x'] for x in sampled_episodes]).reshape(1,-1).T, np.hstack([x['a'] for x in sampled_episodes]).reshape(1,-1).T]), axis=0))
                    sampled_episodes.pop()
                    if (float(num_unique)/maximum) >= percentage:
                        new_episode = {k:val[:max(1,(idx-1))] for k,val in episode.iteritems()}
                        sampled_episodes.append(new_episode)
                        num_unique = len(np.unique(np.hstack([np.hstack([x['x'] for x in sampled_episodes]).reshape(1,-1).T, np.hstack([x['a'] for x in sampled_episodes]).reshape(1,-1).T]), axis=0))
                        break
                done = True
            else:
                pass

    dataset.episodes = sampled_episodes
    dataset['x'] = np.hstack([x['x'] for x in dataset.episodes])
    dataset['a'] = np.hstack([x['a'] for x in dataset.episodes])
    dataset['x_prime'] = np.hstack([x['x_prime'] for x in dataset.episodes])
    dataset['cost'] = np.hstack([x['cost'] for x in dataset.episodes])
    dataset['done'] = np.hstack([x['done'] for x in dataset.episodes])

    print 'Number of Episodes: ', len(sampled_episodes)
    print 'Num unique: ', num_unique
    print 'Percentage of data: ', float(num_unique)/maximum
    # print np.unique(np.hstack([np.hstack([x['x'] for x in sampled_episodes]).reshape(1,-1).T, np.hstack([x['a'] for x in sampled_episodes]).reshape(1,-1).T]), axis=0)

    # Importance Sampling
    approx_ips, exact_ips, approx_pdis, exact_pdis, dr, wdr, am = ips.run(dataset, policy, policy_old, epsilon, gamma)
    # approx_ips, exact_ips, approx_pdis, exact_pdis, dr, wdr, am = 0,0,0,0,0,0,0
    # FQE
    
    # evaluated = fqe.run(policy, 'g', dataset, epochs=1000, epsilon=1e-8, desc='FQE epsilon %s' % np.round(epsilon,2),position_of_holes=position_of_holes, position_of_goals=position_of_goals, g_idx=0)
    evaluated = FQE(dataset, policy)



    # evaluated = 0
    print exact-exact, evaluated-exact, approx_ips-exact, exact_ips-exact, approx_pdis-exact, exact_pdis-exact, dr-exact, wdr-exact, am-exact
    dataset.episodes = all_episodes
    dataset['x'] = np.hstack([x['x'] for x in dataset.episodes])
    dataset['a'] = np.hstack([x['a'] for x in dataset.episodes])
    dataset['x_prime'] = np.hstack([x['x_prime'] for x in dataset.episodes])
    dataset['cost'] = np.hstack([x['cost'] for x in dataset.episodes])
    dataset['done'] = np.hstack([x['done'] for x in dataset.episodes])
    return exact-exact, evaluated-exact, approx_ips-exact, exact_ips-exact, approx_pdis-exact, exact_pdis-exact, dr-exact, wdr-exact, am-exact

def FQE(dataset, policy, gamma=.9, epsilon=0.001):
    # U1 = {}
    # for x in range(64):
    #     U1[x] = {}
    #     for y in range(4):
    #         U1[x][y] = 0

    U1 = np.random.uniform(size=(64,4))*3.
    data = np.hstack([np.hstack([x['x'] for x in dataset.episodes]).reshape(1,-1).T, np.hstack([x['a'] for x in dataset.episodes]).reshape(1,-1).T, np.hstack([x['x_prime'] for x in dataset.episodes]).reshape(1,-1).T, np.hstack([x['cost'] for x in dataset.episodes]).reshape(1,-1).T, np.hstack([x['done'] for x in dataset.episodes]).reshape(1,-1).T])
    data = np.unique(data, axis=0)
    pi_of_x_prime = policy(data[:,2])
    data = np.hstack([data, pi_of_x_prime.reshape(1,-1).T]).astype(int)
    print 'Num unique in FQE: ', data.shape[0]
    
    while True:
        U = U1.copy()
        delta = 0
        
        for x,a,x_prime,cost,done,new_a in data:
            U1[x, a] = cost + gamma * U1[x_prime, new_a]*(1-done)

            delta = max(delta, abs(U1[x,a] - U[x,a]))

        if delta < epsilon * (1 - gamma) / gamma:
             return U[0,policy([0])][0]

def get_dataset(N, epsilon, exact_evaluation):
    num_goal = 0
    num_hole = 0
    dataset = Dataset(1, (1,), (3,))
    policy_old.Q.model = load_model(old_policy_path)
    for i in tqdm(range(N)):
        x = env.reset()
        dataset.start_new_episode(x)
        done = False
        time_steps = 0
        while not done:
            time_steps += 1
            
            if policy_old is not None:
                action = policy_old([x])[0]
                if np.random.random() < epsilon:
                    action = np.random.randint(action_space_dim)
            else:
                action = np.random.randint(action_space_dim)
            x_prime , reward, done, _ = env.step(action)
            c,g = reward
            g= g[0]

            if c: num_goal += 1
            if g: num_hole += 1
            c = c
            g = [g, 0]
            
            dataset.append(  action, x_prime, np.hstack([c,g]), done )

                # x,
                #              action,
                #              x_prime,
                #              c,
                #              g,
                #              done) #{(x,a,x',c(x,a), g(x,a)^T, done)}


            x = x_prime
        # print 'Epoch: %s. Num steps: %s. Avg episode length: %s' % (i, time_steps, float(len(problem.dataset)/(i+1)))
    dataset.preprocess('lake')
    dataset['x'] = dataset['frames'][dataset['prev_states']]
    dataset['x_prime'] = dataset['frames'][dataset['next_states']]

    print 'Epsilon %s. Number goals: %s. Number holes: %s.' % (epsilon, num_goal, num_hole)
    print 'Distribution:' 
    print np.histogram(dataset['x_prime'], bins=np.arange(map_size[0]*map_size[1]+1)-.5)[0].reshape(map_size)
    print len(dataset['x'])
    print len(np.unique(np.hstack([dataset['x'].reshape(1,-1).T,  dataset['a'].reshape(1,-1).T, dataset['x_prime'].reshape(1,-1).T ]), axis=0))
    
    which = 'g'
    dataset.set_cost(which,0)

    # Exact
    exact_c, exact_g, _ = exact_evaluation.run(policy)
    if which == 'g':
        exact = exact_g
    else:
        exact = exact_c
    exact = exact[0]
    
    dones = np.hstack([0,1+np.where(dataset['done'])[0]])
    dataset.buffer_episodes = dataset.episodes
    dataset.episodes = []
    dataset['x'] = dataset['x'].reshape(-1) 
    dataset['a'] = dataset['a'].reshape(-1) 
    dataset['x_prime'] = dataset['x_prime'].reshape(-1) 
    dataset['cost'] = dataset['cost'].reshape(-1) 
    dataset['done'] = dataset['done'].reshape(-1) 

    for low_, high_ in tqdm(zip(dones[:-1], dones[1:])):
        new_episode ={
            'x': dataset['x'][low_:high_].reshape(-1),
            'a': dataset['a'][low_:high_].reshape(-1),
            'x_prime': dataset['x_prime'][low_:high_].reshape(-1),
            'cost': dataset['cost'][low_:high_].reshape(-1),
            'done': dataset['done'][low_:high_].reshape(-1),
        }
        assert new_episode['done'][0] == 0
        assert new_episode['done'][-1] == 1
        assert sum(new_episode['done']) == 1

        dataset.episodes.append(new_episode)

    return dataset, exact

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

main(policy_old, policy, model_type=testing_model_type)
path = os.path.join(os.getcwd(), 'experimental_results')
files = os.listdir(path)
csvs = [f for f in files if 'fqe_quality' in f]
tmp = pd.DataFrame([csv.split('.csv')[0].split('_')[2:] for csv in csvs], columns=['year','month','day','hour','minute','a','b'])
results_file = 'fqe_quality_' + '_'.join(tmp.sort_values(by=['year','month','day','hour','minute'], ascending=False).iloc[0]) + '.csv'
df = pd.read_csv(os.path.join(path, results_file))
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
    for i, col in enumerate(['fqe', 'approx_ips', 'doubly_robust']):
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
