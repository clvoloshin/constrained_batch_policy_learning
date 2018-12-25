"""
Created on December 13, 2018

@author: clvoloshin, 
"""


import numpy as np
import scipy.signal as signal




class ExactPolicyEvaluator(object):
    def __init__(self, action_space_map, gamma, env=None):
        '''
        An implementation of Exact Policy Evaluation through Monte Carlo

        In this case since the environment is fixed and initial states are fixed
        then this will be exact
        '''
        self.gamma = gamma
        self.action_space_map = action_space_map
        # self.initial_states = initial_states
        # self.state_space_dim = state_space_dim
        if env is not None:
            self.env = env
        else:
            raise

    def run(self, policy, *args, **kw):

        environment_is_dynamic = not self.env.deterministic

        if 'policy_is_greedy' not in kw:
            kw['policy_is_greedy']=True
            policy_is_greedy=True
        else:
            policy_is_greedy= kw['policy_is_greedy']
        
        if not isinstance(policy,(list,)):
            policy = [policy]


        if not environment_is_dynamic and policy_is_greedy:
            c,g = self.determinstic_env_and_greedy_policy(policy, **kw)
            if len(args) > 0:
                if args[0] == 'c':
                    return c
                else:
                    try:
                        return g[i]
                    except:
                        if isinstance(g,(list,)) and len(g) > 1:
                            assert False, 'Index error'
                        else:
                            return g
            else:
                return c,g

        else:
            return self.stochastic_env_or_policy(policy, **kw)

    def get_Qs(self, policy, initial_states, state_space_dim, idx=0):
        Q = []
        for initial_state in initial_states:
            self.env.isd = np.eye(state_space_dim)[initial_state]

            if not isinstance(policy,(list,)):
                policy = [policy]
            Q.append(self.determinstic_env_and_greedy_policy(policy, render=False, verbose=False)[idx])
        
        self.env.isd = np.eye(state_space_dim)[0]
        return Q

    def stochastic_env_or_policy(self, policy, render=False, verbose=False, **kw):
        '''
        Run the evaluator
        '''

        all_c = []
        all_g = []
        if len(policy) > 1: import pdb; pdb.set_trace()
        for pi in policy:
            trial_c = []
            trial_g = []
            for i in range(3):
                c = []
                g = []
                x = self.env.reset()
                done = False
                time_steps = 0
                
                while not done:
                    time_steps += 1
                    if render and (i == 0): self.env.render()
                    
                    action = pi([x])[0]

                    x_prime , cost, done, _ = self.env.step(self.action_space_map[action])

                    done = self.env.is_early_episode_termination(time_steps)
                    if verbose: print x,action,x_prime,cost
                    
                    c.append(cost[0])
                    g.append(cost[1])

                    x = x_prime
                trial_c.append(c)
                trial_g.append(g)

            all_c.append(np.mean([self.discounted_sum(x, self.gamma) for x in trial_c]))
            all_g.append(np.mean([self.discounted_sum(np.array(x), self.gamma) for x in trial_g], axis=0).tolist())
            # all_g.append(np.mean([self.discounted_sum(x, self.gamma) for x in trial_g]))
        
        c = np.mean(all_c, axis=0)
        g = np.mean(all_g, axis=0)

        return c,g


    def determinstic_env_and_greedy_policy(self, policy, render=False, verbose=False, **kw):
        '''
        Run the evaluator
        '''

        all_c = []
        all_g = []
        for pi in policy:
            c = []
            g = []
            x = self.env.reset()
            if render: self.env.render()
            done = False
            time_steps = 0
            
            while not done:
                time_steps += 1
                
                
                action = pi([x])[0]

                x_prime , cost, done, _ = self.env.step(self.action_space_map[action])

                done = done or self.env.is_early_episode_termination(time_steps)

                if verbose: print x,action,x_prime,cost
                if render: self.env.render()
                c.append(cost[0])
                g.append(cost[1])

                x = x_prime
            all_c.append(c)
            all_g.append(g)

        
        c = np.mean([self.discounted_sum(x, self.gamma) for x in all_c])
        g = np.mean([self.discounted_sum(np.array(x), self.gamma) for x in all_g], axis=0).tolist()

        if not isinstance(g,(list,)):
            g = [g]

        return c,g

    @staticmethod
    def discounted_sum(costs, discount):
        '''
        Calculate discounted sum of costs
        '''
        y = signal.lfilter([1], [1, -discount], x=costs[::-1])
        return y[::-1][0]
        
        


