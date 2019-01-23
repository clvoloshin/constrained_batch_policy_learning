# from pyvirtualdisplay import Display
# display = Display(visible=0, size=(1280, 1024))
# display.start()
import deepdish as dd
from replay_buffer import Dataset
from config_car import *
import os
import numpy as np
import scipy.signal as signal
from env_nn import CarNN
from keras.models import load_model
np.random.seed(2718)

which_pi = './videos/ohio/run_1/pi_1.hdf5'
directory = 'seed_2_data'
action_data = dd.io.load(os.path.join(os.getcwd(), directory, 'car_data_actions_seed_2.h5'))
frame_data = dd.io.load(os.path.join(os.getcwd(), directory, 'car_data_frames_seed_2.h5'))
done_data = dd.io.load(os.path.join(os.getcwd(), directory, 'car_data_is_done_seed_2.h5'))
next_state_data = dd.io.load(os.path.join(os.getcwd(), directory, 'car_data_next_states_seed_2.h5'))
current_state_data = dd.io.load(os.path.join(os.getcwd(), directory, 'car_data_prev_states_seed_2.h5'))
cost_data = dd.io.load(os.path.join(os.getcwd(), directory, 'car_data_rewards_seed_2.h5'))


frame_gray_scale = np.zeros((len(frame_data),96,96)).astype('float32')
for i in range(len(frame_data)):
    frame_gray_scale[i,:,:] = np.dot(frame_data[i,:,:,:]/255. , [0.299, 0.587, 0.114])

dic = {'frames':frame_gray_scale,
            'prev_states': current_state_data,
            'next_states': next_state_data,
            'a': action_data,
            'c':cost_data[:,0],
            'g':cost_data[:,1:],
            'done': done_data
            }

data = Dataset(num_frame_stack, pic_size, (len(constraints) + 1,) )
data.data = dic  
EVALUATING = 'c'


def sample_N_trajectories(dataset, N):
    dones = np.where(dataset['done'])[0]
    dones = np.hstack([[0], dones])
    trajectory_idxs = zip(dones[:-1], dones[1:])
    N = min(len(trajectory_idxs), N)
    idxs = np.random.choice(len(trajectory_idxs), size=N, replace=False)
    return np.array(trajectory_idxs)[idxs]


def create_trajectories(dataset, N):
    idxs = sample_N_trajectories(dataset, N)
    episodes = []
    for low, high in idxs:
        x = np.rollaxis(dataset['frames'][dataset['prev_states'][low:high]],1,4)
        actions = np.atleast_2d(dataset['a'][low:high]).T
        x_prime = np.rollaxis(dataset['frames'][dataset['next_states'][low:high]],1,4)
        dataset_costs = dataset[EVALUATING][low:high]
        dones = dataset['done'][low:high]
        episode = {
                    'x': x,
                    'a': actions,
                    'x_prime': x_prime,
                    'cost': dataset_costs,
                    'done': dones,
                    }
        episodes.append(episode)
    return episodes

def pdis(episodes, pi_new, pi_old, gamma):
    '''
    Per decision importance sampling

    sum_{t=1}^{max L} gamma^t  1/n sum_{i=1}^n (PI_{tau=1}^t p_new/p_old) R^i_t
    '''
    values = []
    for episode in episodes:

        numerator = pi_new.all_actions([episode['x']], x_preprocessed=True)[np.arange(len(episode['a'])), episode['a'].reshape(-1)]
        denominator = pi_old.all_actions([episode['x']], x_preprocessed=True)[np.arange(len(episode['a'])), episode['a'].reshape(-1)]
        importance_weight = np.cumprod(numerator/denominator)

        values.append( discounted_sum(importance_weight * episode['cost'], gamma) )

    return np.mean(values)

def WDR(episodes, pi_new, pi_old, gamma):
    # \hat{v}^pi(s) = \sum_t gamma^t * \hat{r}^pi(s,t) 
    #               = \sum_t * \sum_a pi(a|s) \hat{r}^pi(s,a,t)
    #               = \sum_t * \hat{r}^\pi (s, A, t) where A = argmin_a pi(s), since our pi_new is deterministic

    # WDR = 1/n \sum_i \hat{v}^\pi_new (S_0^{H_i}) 
    #            + \sum_i \sum_t gamma^t w_t^i [R_t^{H_i} + gamma \hat{v}^\pi_new (S_{t+1}^{H_i}) - \hat{q}(S_t^{H_i}, A_t^{H_i})]

    # since pi_new, pi_old, ..etc, deterministic then:
    # Thus, WDR = \hat{v}^\pi_new (S_0) + \sum_i \sum_t gamma^t w_t^i [R_t^{H_i} - \hat{r}(S_t^{H_i},A_t^{H_i},0)]
    # 
    # w_t^i = p_t^i / sum_{j=1}^n p_i^j
    #
    # p_t^i = prod_{i=0}^t pi_new(A_i|S_i) / pi_old(A_i|S_i) 

def discounted_sum(costs, discount):
    '''
    Calculate discounted sum of costs
    '''
    y = signal.lfilter([1], [1, -discount], x=costs[::-1])
    return y[::-1][0]


def main():
    episodes = create_trajectories(data, 50)

    for episode in tqdm(episodes):
        numerator = pi_new.all_actions([episode['x']], x_preprocessed=True)[np.arange(len(episode['a'])), episode['a'].reshape(-1)]
        denominator = pi_old.all_actions([episode['x']], x_preprocessed=True)[np.arange(len(episode['a'])), episode['a'].reshape(-1)]
    
    model = load_model(which_pi)
    pi_new = CarNN(state_space_dim, 
                  action_space_dim, 
                  max_Q_fitting_epochs, 
                  gamma, 
                  model_type=model_type,
                  num_frame_stack=num_frame_stack)
    pi_new.model.set_weights(model.get_weights())

    pdis_output = pdis(episodes, pi_new, pi_new, gamma)
    import pdb; pdb.set_trace()

main()




