


#### Setup Gym 
import gym
import numpy as np
from gym.envs.registration import register
from gym.envs.box2d.car_racing import *
import itertools


class ExtendedCarRacing(CarRacing):
    def __init__(self, init_seed, stochastic, max_pos_costs):
        super(ExtendedCarRacing, self).__init__()
        self.deterministic = not stochastic
        self.init_seed = init_seed
        self.max_pos_costs = max_pos_costs
        self.min_cost = -1000. # defined by CarRacing env. In fact, this is only the minimum if you can instantaneously do the whole track

    def is_early_episode_termination(self, cost):
        if cost > 0:
            self.pos_cost_counter += 1
            done = (self.pos_cost_counter > self.max_pos_costs)
        else:
            self.pos_cost_counter = 0
            done = False

        return done

    def reset(self):
        self._destroy()
        self.amount_of_time_spent_doing_nothing = 0
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.pos_cost_counter = 0
        self.road_poly = []
        self.human_render = False
        if self.deterministic:
            st0 = np.random.get_state()
            self.seed(self.init_seed)

        while True:
            success = self._create_track()
            if success: break
            print("retry to generate track (normal if there are not many of this messages)")
        self.car = Car(self.world, *self.track[0][1:4])

        if self.deterministic:
            # set seed back after recreating same track
            np.random.set_state(st0) 
        return self.step(None)[0]

    def step(self, action):
        if action is not None:
            self.car.steer(-action[0])
            self.car.gas(action[1])
            self.car.brake(action[2])

        self.car.step(1.0/FPS)
        self.world.Step(1.0/FPS, 6*30, 2*30)
        self.t += 1.0/FPS

        self.state = self.render("state_pixels")

        step_reward = 0
        c = 0
        g = [0.]
        
        done = False
        if action is not None: # First step without action, called from reset()
            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            #self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward

            if self.tile_visited_count==len(self.track):
                done = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                done = True
                step_reward = -100

            c = -step_reward
            g = [0.]


        return self.state, (c,g), done, {}

# env = gym.make('CarRacing-v0')
init_seed = 0
stochastic_env = False # deterministic
max_pos_costs = 12 # The maximum allowable positive cost before ending episode early
max_time_spent_in_episode = 2000
env = ExtendedCarRacing(init_seed, stochastic_env, max_pos_costs)

#### Hyperparam
gamma = 0.95
max_epochs = 2 # max number of epochs over which to collect data
max_Q_fitting_epochs = 1 #max number of epochs over which to converge to Q^\ast.   Fitted Q Iter
max_eval_fitting_epochs = 1 #max number of epochs over which to converge to Q^\pi. Off Policy Eval
lambda_bound = 30. # l1 bound on lagrange multipliers
epsilon = .01 # termination condition for two-player game
deviation_from_old_policy_eps = .95 #With what probabaility to deviate from the old policy
# convergence_epsilon = 1e-6 # termination condition for model convergence
# action_space_dim = env.nA # action space dimension
# state_space_dim = env.nS # state space dimension
eta = 50. # param for exponentiated gradient algorithm
# initial_states = [[0]] #The only initial state is [1,0...,0]. In general, this should be a list of initial states
# non_terminal_states = np.nonzero(((env.desc == 'S') + (env.desc == 'F')).reshape(-1))[0] # Used for dynamic programming. this is an optimization to make the algorithm run faster. In general, you may not have this
max_number_of_main_algo_iterations = 100 # After how many iterations to cut off the main algorithm
model_type = 'cnn'
old_policy_name = 'pi_old_car_{0}.h5'.format(model_type)


## DQN Param
num_iterations = 3000
sample_every_N_transitions = 12
batchsize = 64
copy_over_target_every_M_training_iterations = 1000
buffer_size = 10000
min_epsilon = .02
initial_epsilon = .3 
epsilon_decay_steps = num_iterations


# Other

state_space_dim = (96,96,1)

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


