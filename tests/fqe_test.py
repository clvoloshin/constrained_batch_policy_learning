from pyvirtualdisplay import Display
display = Display(visible=1, size=(1280, 1024))
display.start()
from fitted_off_policy_evaluation import CarFittedQEvaluation
from exact_policy_evaluation import ExactPolicyEvaluator
from config_car import *
from fitted_algo import FittedAlgo
import numpy as np
from tqdm import tqdm
from env_nn import *
from thread_safe import threadsafe_generator
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from env_dqns import CarDQN
import deepdish as dd
from keras.models import load_model
import time
from replay_buffer import Dataset
from stochastic_policy import StochasticPolicy


model_dir = os.path.join(os.getcwd(), 'models')
old_policy_path = os.path.join(model_dir, old_policy_name)
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
                    frame_skip = frame_skip,
                    pic_size = pic_size,
                    models_path = os.path.join(model_dir,'weights.{epoch:02d}-{loss:.2f}.hdf5'),
                    )
policy_old.Q.model = load_model(old_policy_path)
policy_old.Q.all_actions_func = K.function([policy_old.Q.model.get_layer('inp').input], [policy_old.Q.model.get_layer('all_actions').output])
print 'Exact Evaluation: '
exact_policy_algorithm = ExactPolicyEvaluator(action_space_map, gamma, env=env, frame_skip=frame_skip, num_frame_stack=num_frame_stack, pic_size = pic_size, constraint_thresholds=constraint_thresholds, constraints_cared_about=constraints_cared_about)
#policy_old.Q.evaluate(render=True, environment_is_dynamic=False, to_monitor=True)
print exact_policy_algorithm.run(policy_old.Q, to_monitor=False)


# policy_to_test = StochasticPolicy(policy_old, action_space_dim, exact_policy_algorithm, epsilon=0., prob=prob)

tic = time.time()
action_data = dd.io.load('./seed_2/car_data_actions_seed_2.h5')
frame_data = dd.io.load('./seed_2/car_data_frames_seed_2.h5')
done_data = dd.io.load('./seed_2/car_data_is_done_seed_2.h5')
next_state_data = dd.io.load('./seed_2/car_data_next_states_seed_2.h5')
current_state_data = dd.io.load('./seed_2/car_data_prev_states_seed_2.h5')
cost_data = dd.io.load('./seed_2/car_data_rewards_seed_2.h5')


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

data.data['g'] = data.data['g'][:,constraints_cared_about]
data.data['g'] = (data.data['g'] >= constraint_thresholds[:-1]).astype(int)   

FQE = CarFittedQEvaluation(state_space_dim, action_space_dim, max_eval_fitting_epochs, gamma, model_type=model_type,num_frame_stack=num_frame_stack)


FQE.run(policy_old.Q,'c', data, desc='FQE C', g_idx=1, testing=True, epochs=1)


def rolling_sum(a, n=4) : ret = np.cumsum(a, axis=1, dtype=float); ret[:, n:] = ret[:, n:] - ret[:, :-n]; return ret[:, n - 1:];