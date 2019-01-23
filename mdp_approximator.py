
import numpy as np
import keras
from keras.models import Sequential, Model as KerasModel
from keras.layers import Input, Dense, Flatten, concatenate, dot, MaxPooling2D
from keras.losses import mean_squared_error
import scipy.signal as signal
from env_nn import LakeNN
from keras import optimizers

import gym



class MDPApproximator(LakeNN):
    def __init__(self, env, *args, **kw):
        '''
        Approximate P(s'| s,a)
        '''
        self.env = env

        self.model_type = kw['model_type'] if 'model_type' in kw else 'mlp'
        self.gamma = .9
        super(MDPApproximator, self).__init__(68, 1, [8,8], 4, self.gamma, convergence_of_model_epsilon=1e-10, model_type='mlp', num_frame_stack=(1,), frame_skip=1, pic_size = (1,))
        self.create_model(68,1)

    def create_model(self, num_inputs, num_outputs):
        if self.model_type == 'mlp':
            model = Sequential()
            def init(): return keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=np.random.randint(2**32))
            model.add(Dense(64, activation='tanh', input_shape=(num_inputs,),kernel_initializer=init(), bias_initializer=init()))
            model.add(Dense(num_outputs, activation='linear',kernel_initializer=init(), bias_initializer=init()))
            model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])
            self.model = model
        else:
            self.model = super(MDPApproximator, self).create_model(num_inputs, num_outputs)

    def run(self, dataset):
        '''
        probability of
        transitioning from s to s'
        given action a is the number of
        times this transition was observed divided by the number
        of times action a was taken in state s. If D contains no examples
        of action a being taken in state s, then we assume
        that taking action a in state s always causes a transition to
        the terminal absorbing state.

        Since everything is deterministic then P(s'|s,a) = 0 or 1.
        '''
        transitions = np.vstack([dataset['x'],dataset['a'],dataset['x_prime']]).T
        unique, idx, count = np.unique(transitions, return_index=True, return_counts=True, axis=0)

        partial_transitions = np.vstack([dataset['x'],dataset['a']]).T
        unique_a_given_x, idx_a_given_x, count_a_given_x = np.unique(partial_transitions, return_index=True, return_counts=True, axis=0)

        # key=(state, action). value= number of times a was taking in state
        all_counts_a_given_x = {tuple(key):value for key,value in zip(unique_a_given_x,count_a_given_x)}
        
        prob = {}
        for idx,row in enumerate(unique): 
            if tuple(row[:-1]) in prob:
                prob[tuple(row[:-1])][row[-1]] = count[idx] / all_counts_a_given_x[(row[0],row[1])]
            else:
                prob[tuple(row[:-1])] = {}
                prob[tuple(row[:-1])][row[-1]] = count[idx] / all_counts_a_given_x[(row[0],row[1])]

        all_transitions = np.vstack([dataset['x'],dataset['a'],dataset['x_prime'], dataset['done']]).T
        self.terminal_transitions = {tuple([x,a,x_prime]):1 for x,a,x_prime in all_transitions[all_transitions[:,-1] == True][:,:-1]}

        # Actually fitting R, not Q_k
        self.Q_k = self.model #init_Q(model_type=self.model_type)
        X_a = np.array(zip(dataset['x'],dataset['a']))#dataset['state_action']
        x_prime = dataset['x_prime']
        index_of_skim = self.skim(X_a, x_prime)
        self.fit(X_a[index_of_skim], dataset['cost'][index_of_skim], batch_size=len(index_of_skim), verbose=0, epochs=1000)
        self.reward = self
        self.P = prob

    def skim(self, X_a, x_prime):
        full_set = np.hstack([X_a, x_prime.reshape(1,-1).T])
        idxs = np.unique(full_set, axis=0, return_index=True)[1]
        return idxs

    def R(self, *args):
        # Exact R
        mapping = {0:[0,-1], 1:[1,0], 2:[0,1], 3:[-1,0]}
        x = args[0]
        x, y = np.where(np.arange(np.prod(self.env.desc.shape)).reshape(self.env.desc.shape) == x)
        x,y = x[0], y[0]
        delta_x,delta_y = mapping[args[1][0]]
        new_x = x + delta_x
        new_y = y + delta_y
        new_x,new_y = (new_x,new_y) if (0 <= new_x < self.env.desc.shape[0] and 0 <= new_y < self.env.desc.shape[1]) else (x,y)
        return [[1]] if self.env.desc[new_x,new_y]=='H' else [[0]]

        # Approximated Rewards
        # return self.reward(*args)

    def transition(self, x, a):
        # Exact MDP dynamics 
        # mapping = {0:[0,-1], 1:[1,0], 2:[0,1], 3:[-1,0]}
        # x, y = np.where(np.arange(np.prod(self.env.desc.shape)).reshape(self.env.desc.shape) == x)
        # x,y = x[0], y[0]
        # delta_x,delta_y = mapping[a]
        # new_x = x + delta_x
        # new_y = y + delta_y
        # new_x,new_y = (new_x,new_y) if (0 <= new_x < self.env.desc.shape[0] and 0 <= new_y < self.env.desc.shape[1]) else (x,y)
        # done = True if self.env.desc[new_x,new_y]=='H' else False
        # done = done or (True if self.env.desc[new_x,new_y]=='G' else False)
        # return np.arange(np.prod(self.env.desc.shape)).reshape(self.env.desc.shape)[new_x,new_y], done
        
        #Approximated dynamics
        if tuple([x,a]) in self.P:
            try:
                state = np.random.choice(self.P[(x,a)].keys(), p=self.P[(x,a)].values())
            except:
                import pdb; pdb.set_trace()
            done = False
        else:
            state = None
            done = True

        return state, done

    def Q(self, policy, x, a):

        Qs = []

        state = x
        original_a = a
        done = False
        costs = []
        trajectory_length = -1
        # Q
        while not done and trajectory_length < 200:
            trajectory_length += 1
            if trajectory_length > 0:
                a = policy([state])[0]

           
            new_state, done = self.transition(state, a)
            costs.append( self.R([state], [a])[0][0] )
            if (tuple([state,a,new_state]) in self.terminal_transitions):
                done = True
                
            
            state = new_state

        return self.discounted_sum(costs, self.gamma)

    def V(self, policy, x):
        state = x
        done = False
        weighted_costs = []
        trajectory_length = -1
        # V
        while not done and trajectory_length < 200:
            trajectory_length += 1
            # Because greedy deterministic policy
            a = policy([state])[0]

            new_state, done = self.transition(state, a)
            weighted_costs.append( self.R([state], [a])[0][0] )
            if (tuple([state,a,new_state]) in self.terminal_transitions):
                done = True
                
            state = new_state

        return self.discounted_sum(weighted_costs, self.gamma)

    @staticmethod
    def discounted_sum(costs, discount):
        '''
        Calculate discounted sum of costs
        '''
        y = signal.lfilter([1], [1, -discount], x=costs[::-1])
        return y[::-1][0]





