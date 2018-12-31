#### Setup Gym 
from car_racing import ExtendedCarRacing
import itertools

# env = gym.make('CarRacing-v0')
init_seed = 0
stochastic_env = True # = not deterministic
max_pos_costs = 12 # The maximum allowable positive cost before ending episode early
max_time_spent_in_episode = 2000
env = ExtendedCarRacing(init_seed, stochastic_env, max_pos_costs)

#### Hyperparam
gamma = 0.99
max_epochs = 1 # max number of epochs over which to collect data
max_Q_fitting_epochs = 10 #max number of epochs over which to converge to Q^\ast.   Fitted Q Iter
max_eval_fitting_epochs = 1 #max number of epochs over which to converge to Q^\pi. Off Policy Eval
lambda_bound = 30. # l1 bound on lagrange multipliers
epsilon = .01 # termination condition for two-player game
deviation_from_old_policy_eps = 0.0 #With what probabaility to deviate from the old policy
# convergence_epsilon = 1e-6 # termination condition for model convergence
# action_space_dim = env.nA # action space dimension
# state_space_dim = env.nS # state space dimension
eta = 50. # param for exponentiated gradient algorithm
# initial_states = [[0]] #The only initial state is [1,0...,0]. In general, this should be a list of initial states
# non_terminal_states = np.nonzero(((env.desc == 'S') + (env.desc == 'F')).reshape(-1))[0] # Used for dynamic programming. this is an optimization to make the algorithm run faster. In general, you may not have this
max_number_of_main_algo_iterations = 100 # After how many iterations to cut off the main algorithm
model_type = 'cnn'
old_policy_name = 'pi_old_car_{0}.hdf5'.format(model_type)


# Constraint 1: We'd like the number of times you brake to be less than 10% of the time 
# Constraint 2: We'd like the car to stay within 15 units of the center of the track 90% of the time 
constraint_thresholds = [1., 15.] + [1]
constraints_cared_about = [-1,2]
constraints = [.1, .1] + [0]

## DQN Param
num_iterations = 5000
sample_every_N_transitions = 4
batchsize = 64
copy_over_target_every_M_training_iterations = 250
buffer_size = 20000
min_epsilon = .1
initial_epsilon = 1.
epsilon_decay_steps = 4000 #num_iterations
num_frame_stack=3
min_buffer_size_to_train = 5000
frame_skip=3
pic_size = (96, 96, 3)

# Other

state_space_dim = (96, 96, num_frame_stack)

# action_space_map = { 
#                 0: [0.0,  0.0,  0.0],   # Brake
#                 1: [-0.6, 0.05, 0.0],   # Sharp left
#                 2: [0.6,  0.05, 0.0],   # Sharp right
#                 3: [0.0,  0.3,  0.0]  } # Staight

action_space_map = {}
for i, action in enumerate([k for k in itertools.product([-1, 0, 1], [1, 0], [0.2, 0])]):
    action_space_map[i] = action

action_space_dim = len(action_space_map)
prob = [1/float(action_space_dim)]*action_space_dim # Probability with which to explore space when deviating from old policy


