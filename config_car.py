


#### Setup Gym 
import gym
import numpy as np
from gym.envs.registration import register
from gym.envs.box2d.car_racing import *



class ExtendedCarRacing(CarRacing):
    def __init__(self, init_seed, deterministic, max_doing_nothing, max_episode_length):
        super(ExtendedCarRacing, self).__init__()
        self.deterministic = deterministic
        self.init_seed = init_seed
        self.max_doing_nothing = max_doing_nothing
        self.max_episode_length = max_episode_length
        self.min_cost = -1000. # defined by CarRacing env. In fact, this is only the minimum if you can instantaneously do the whole track

    def is_early_episode_termination(self, episode_length):
        true_speed = np.sqrt(np.square(self.car.hull.linearVelocity[0]) + np.square(self.car.hull.linearVelocity[1]))
        if true_speed < 1e-1:
            self.amount_of_time_spent_doing_nothing += 1
        else:
            self.amount_of_time_spent_doing_nothing = 0

        if (self.amount_of_time_spent_doing_nothing > self.max_doing_nothing) or (episode_length > self.max_episode_length):
            return True
        else:
            return False

    def reset(self):
        self._destroy()
        self.amount_of_time_spent_doing_nothing = 0
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []
        self.human_render = False
        if self.deterministic:
            st0 = np.random.get_state()
            self.seed(init_seed)

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
deterministic = True
max_doing_nothing = 100 # 100 frames = 2 seconds
max_time_spent_in_episode = 2000
env = ExtendedCarRacing(init_seed, deterministic, max_doing_nothing, max_time_spent_in_episode)

#### Hyperparam
gamma = 0.99
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
prob = [1/4.]*4 # Probability with which to explore space when deviating from old policy
model_type = 'cnn'
old_policy_name = 'pi_old_car_{0}.h5'.format(model_type)



## DQN Param
num_iterations = 5000
sample_every_N_transitions = 100
batchsize = 10000
copy_over_target_every_M_training_iterations = 1000
buffer_size = 100000

# Other

state_space_dim = env.observation_space.high.shape
stochastic_env = True
action_space_map = { 
                0: [0.0,  0.0,  0.0],   # Brake
                1: [-0.6, 0.05, 0.0],   # Sharp left
                2: [0.6,  0.05, 0.0],   # Sharp right
                3: [0.0,  0.3,  0.0]  } # Staight

action_space_dim = len(action_space_map)
