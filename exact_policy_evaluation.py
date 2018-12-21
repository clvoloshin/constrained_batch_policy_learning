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

    def run(self, policy, *args, **kw):

        if 'environment_is_dynamic' not in kw:
            kw['environment_is_dynamic']=False
            environment_is_dynamic=False
        else:
            environment_is_dynamic = True 

        if 'policy_is_greedy' not in kw:
            kw['policy_is_greedy']=True
            policy_is_greedy=True
        else:
            policy_is_greedy= False
        
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

        elif not environment_is_dynamic:
            return self.determinstic_env_and_stochastic_policy(policy, **kw)
        else:
            raise NotImplemented

    def get_Qs(self, policy, idx=0):
        Q = []
        for initial_state in self.initial_states:
            self.env.isd = np.eye(self.state_space_dim)[initial_state]

            if not isinstance(policy,(list,)):
                policy = [policy]
            Q.append(self.determinstic_env_and_greedy_policy(policy, render=False, verbose=False)[idx])
        
        self.env.isd = np.eye(self.state_space_dim)[0]
        return Q

    def determinstic_env_and_stochastic_policy(self, policy, render=False, verbose=False, **kw):
        '''
        Run the evaluator
        '''

        all_c = []
        all_g = []
        if len(policy) > 1: import pdb; pdb.set_trace()
        for pi in policy:
            trial_c = []
            trial_g = []
            for i in range(5000):
                c = []
                g = []
                x = self.env.reset()
                if render: self.env.render()
                done = False
                time_steps = 0
                
                while not done and time_steps < 100:
                    time_steps += 1
                    
                    
                    action = pi([x])[0]

                    x_prime , reward, done, _ = self.env.step(action)

                    if verbose: print x,action,x_prime,reward, int(done and not reward)
                    if render: self.env.render()
                    c.append(-reward)
                    g.append(done and not reward)

                    x = x_prime
                trial_c.append(c)
                trial_g.append(g)

            all_c.append(np.mean([self.discounted_sum(x, self.gamma) for x in trial_c]))
            all_g.append(np.mean([self.discounted_sum(x, self.gamma) for x in trial_g]))

        c = np.mean(all_c)
        g = np.mean(all_g)

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
            states_seen = {}
            x = self.env.reset()
            if render: self.env.render()
            states_seen[x] = 0
            done = False
            time_steps = 0
            
            while not done:
                time_steps += 1
                
                
                action = pi([x])[0]

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


        c = np.mean([self.discounted_sum(x, self.gamma) for x in all_c])
        g = np.mean([self.discounted_sum(x, self.gamma) for x in all_g])

        
        return c,g

    @staticmethod
    def discounted_sum(costs, discount):
        '''
        Calculate discounted sum of costs
        '''
        y = signal.lfilter([1], [1, -discount], x=costs[::-1])
        return y[::-1][0]
        
        


