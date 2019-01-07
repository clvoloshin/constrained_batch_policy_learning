from pyvirtualdisplay import Display
display = Display(visible=0, size=(1280, 1024))
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

data = {'frames':frame_gray_scale,
            'prev_states': current_state_data,
            'next_states': next_state_data,
            'a': action_data,
            'c':cost_data[:,0],
            'g':cost_data[:,1:],
            'done': done_data
            }

FQE = CarFittedQEvaluation(state_space_dim, 
                     action_space_dim, 
                     max_eval_fitting_epochs, 
                     gamma, 
                     model_type=model_type,
                     num_frame_stack=num_frame_stack)

print 'Exact Evaluation: '
print exact_policy_algorithm.run(policy_old)

FQE.run(policy_old,'c', data, desc='FQE C'% i, g_idx=i, testing=True)






