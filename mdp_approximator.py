"""
Created on December 22, 2018

@author: clvoloshin, 
"""

import numpy as np
from fitted_algo import FittedAlgo
import scipy.signal as signal


import gym



class MDPApproximator(FittedAlgo):
    def __init__(self, *args, **kw):
        '''
        Approximate P(s'| s,a)
        '''
        self.env = gym.make('FrozenLake-no-slip-v0')

        self.model_type = kw['model_type'] if 'model_type' in kw else 'mlp'
        super(MDPApproximator, self).__init__(*args)

    def run(self, dataset):
        '''
        probability of
        transitioning from s to s
        given action a is the number of
        times this transition was observed divided by the number
        of times action a was taken in state s. If D contains no examples
        of action a being taken in state s, then we assume
        that taking action a in state s always causes a transition to
        the terminal absorbing state.
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

        # Actually fitting R, not Q_k
        self.Q_k = self.init_Q(model_type=self.model_type)
        X_a = dataset['state_action']
        x_prime = dataset['x_prime']
        index_of_skim = self.skim(X_a, x_prime)
        self.fit(X_a[index_of_skim],dataset['cost'][index_of_skim], verbose=0, epochs=self.max_epochs)
        self.reward = self.Q_k
        self.P = prob

    def R(self, *args):
        # Exact R
        # mapping = {0:[0,-1], 1:[1,0], 2:[0,1], 3:[-1,0]}
        # x = args[0]
        # x, y = np.where(np.arange(np.prod(self.env.desc.shape)).reshape(self.env.desc.shape) == x)
        # x,y = x[0], y[0]
        # delta_x,delta_y = mapping[args[1][0]]
        # new_x = x + delta_x
        # new_y = y + delta_y
        # new_x,new_y = (new_x,new_y) if (0 <= new_x < self.env.desc.shape[0] and 0 <= new_y < self.env.desc.shape[1]) else (x,y)
        # return [[1]] if self.env.desc[new_x,new_y]=='H' else [[0]]

        # Approximated Rewards
        return self.reward(*args)

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
        
        # Approximated dynamics
        if tuple([x,a]) in self.P:
            state = np.random.choice(self.P[(x,a)].keys(), p=self.P[(x,a)].values())
            done = False
        else:
            state = None
            done = True

        return state, done

    def Q(self, policy, x, a):

        Qs = []
        Vs = []

        state = x
        done = False
        costs = []
        weighted_costs = []
        trajectory_length = -1

        while not done and trajectory_length < 200:
            trajectory_length += 1
            if trajectory_length > 0:
                a = policy([state])[0]

            costs.append( self.R([state], [a])[0][0] )
            # Because greedy deterministic policy
            weighted_costs.append( self.R([state], policy([state]))[0][0] ) 

            
            new_state, done = self.transition(state, a)
            state = new_state

        Q = self.discounted_sum(costs, self.gamma)
        V = self.discounted_sum(weighted_costs, self.gamma)

        return Q, V

    @staticmethod
    def discounted_sum(costs, discount):
        '''
        Calculate discounted sum of costs
        '''
        y = signal.lfilter([1], [1, -discount], x=costs[::-1])
        return y[::-1][0]





