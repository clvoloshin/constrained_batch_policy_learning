
from DQN import DeepQLearning
from env_nn import *

class LakeDQN(DeepQLearning):
    def __init__(self, *args, **kw):
        holes, goals = kw['position_of_holes'], kw['position_of_goals']
        del kw['position_of_holes']
        del kw['position_of_goals']

        super(LakeDQN, self).__init__(*args, **kw)
        
        for key in ['action_space_map','max_time_spent_in_episode','num_iterations','sample_every_N_transitions','batchsize','copy_over_target_every_M_training_iterations', 'buffer_size']:
            if key in kw: del kw[key]

        kw['position_of_holes'],kw['position_of_goals']  = holes, goals
        self.state_space_dim = self.env.nS
        self.action_space_dim = self.env.nA
        self.Q = LakeNN(self.state_space_dim+self.action_space_dim, 1, [self.env.desc.shape[0], self.env.desc.shape[1]], self.action_space_dim, self.gamma, **kw)
        self.Q_target = LakeNN(self.state_space_dim+self.action_space_dim, 1, [self.env.desc.shape[0], self.env.desc.shape[1]], self.action_space_dim, self.gamma, **kw)


class CarDQN(DeepQLearning):
    def __init__(self, *args, **kw):
        
        self.action_space_dim = kw['action_space_dim']
        del kw['action_space_dim']
        super(CarDQN, self).__init__(*args, **kw) 

        for key in ['action_space_map','max_time_spent_in_episode','num_iterations','sample_every_N_transitions','batchsize','copy_over_target_every_M_training_iterations', 'buffer_size']:
            if key in kw: del kw[key]

        self.state_space_dim = self.env.observation_space.high.shape
        self.Q = CarNN(self.state_space_dim, self.action_space_dim, self.gamma, **kw)
        self.Q_target = CarNN(self.state_space_dim, self.action_space_dim, self.gamma, **kw)

        
        






