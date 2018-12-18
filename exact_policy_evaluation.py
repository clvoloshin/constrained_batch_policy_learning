"""
Created on December 13, 2018

@author: clvoloshin, 
"""


import numpy as np
import scipy.signal as signal




class ExactPolicyEvaluator(object):
    def __init__(self, initial_states, state_space_dim, gamma, env=None):
        '''
        An implementation of Exact Policy Evaluation through Monte Carlo

        In this case since the environment is fixed and initial states are fixed
        then this will be exact
        '''
        self.gamma = gamma
        self.initial_states = initial_states
        self.state_space_dim = state_space_dim
        if env is not None:
            self.env = env
        else:
            import gym
            env = gym.make('FrozenLake-no-slip-v0')
            self.env = env

    def run(self, policy, environment_is_dynamic=False, policy_is_greedy=True, **kw):
        
        if not isinstance(policy,(list,)):
            policy = [policy]

        if not environment_is_dynamic and policy_is_greedy:
            return self.determinstic_env_and_greedy_policy(policy, **kw)
        elif not environment_is_dynamic:
            raise NotImplemented
        else:
            raise NotImplemented

    def determinstic_env_and_greedy_policy(self, policy, render=False, verbose=False):
        '''
        Run the evaluator
        '''
        

        use_monte_carlo = len(policy) > 1
        max_epochs = 1000*use_monte_carlo + 1
        number_of_polices = len(policy)

        all_c = []
        all_g = []
        for i in range(max_epochs):
            c = []
            g = []
            states_seen = {}
            x = self.env.reset()
            if render: self.env.render()
            states_seen[x] = 0
            done = False
            time_steps = 0
            while not done:
                time_steps += 1
                
                # Randomly select over convexification. Or deterministically if only 1 policy
                policy_to_use = 0
                if number_of_polices > 1: policy_to_use = np.random.randint(number_of_polices)
                action = policy[policy_to_use]([x])[0]

                x_prime , reward, done, _ = self.env.step(action)

                if verbose: print x,action,x_prime,reward, int(done and not reward)
                if render: self.env.render()
                c.append(-reward)
                g.append(done and not reward)

                '''
                If the policy sends x' -> x_i, a state already seen
                then we have an infinite loop and can terminate and calculate value function
                
                The length of the cycle is the value of time_steps - states_seen[x'].
                If the sum of the costs over this cycle is non-zero then the value function blows up
                for infinite time horizons
                '''
                if x_prime in states_seen:
                    done = True
                    cycle_length = time_steps - states_seen[x_prime]
                    if sum(c[-cycle_length:]) != 0:
                        c.append(np.inf*sum(c[-cycle_length:]))
                    if sum(g[-cycle_length:]) != 0:
                        c.append(np.inf*sum(g[-cycle_length:]))
                else:
                    states_seen[x_prime] = time_steps

                x = x_prime
            all_c.append(c)
            all_g.append(g)

        c = self.discounted_sum(np.mean(np.array(all_c), axis=0), self.gamma)
        g = self.discounted_sum(np.mean(np.array(all_g), axis=0), self.gamma)
        
        return c,g

    @staticmethod
    def discounted_sum(costs, discount):
        '''
        Calculate discounted sum of costs
        '''
        y = signal.lfilter([1], [1, -discount], x=costs[::-1])
        return y[::-1][0]
        
        


