"""
Created on December 13, 2018

@author: clvoloshin, 
"""


import numpy as np
import scipy.signal as signal
from replay_buffer import Buffer
import os


class ExactPolicyEvaluator(object):
    def __init__(self, action_space_map, gamma, env=None, num_frame_stack=None, frame_skip = None, pic_size = None, constraint_thresholds=None, constraints_cared_about=None):
        '''
        An implementation of Exact Policy Evaluation through Monte Carlo

        In this case since the environment is fixed and initial states are fixed
        then this will be exact
        '''
        self.gamma = gamma
        self.action_space_map = action_space_map
        self.constraint_thresholds = constraint_thresholds
        self.constraints_cared_about = constraints_cared_about

        self.num_frame_stack = num_frame_stack 
        self.frame_skip = frame_skip
        self.pic_size = pic_size      
        self.buffer_size = int(2000)
        self.min_buffer_size_to_train = 0                                     
        
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
            for i in range(1):
                c = []
                g = []
                self.buffer = Buffer(num_frame_stack= self.num_frame_stack,buffer_size= self.buffer_size,min_buffer_size_to_train= self.min_buffer_size_to_train,pic_size = self.pic_size,)
                x = self.env.reset()
                self.buffer.start_new_episode(x)
                done = False
                time_steps = 0
                
                while not done:
                    time_steps += 1
                    if (self.env.env_type in ['car']) or render: self.env.render()

                    action = pi([self.buffer.current_state()])[0]

                    cost = []
                    for _ in range(self.frame_skip):
                        x_prime, costs, done, _ = self.env.step(self.action_space_map[action])
                        # if self.render:
                        #     self.env.render()
                        cost.append(costs)
                        if done:
                            break
                    
                    cost = np.vstack([np.hstack(x) for x in cost]).sum(axis=0)
                    if self.constraint_thresholds is not None: 
                        cost[1:][self.constraints_cared_about] = np.array(cost[1:])[self.constraints_cared_about] >= self.constraint_thresholds[:-1]


                    early_done, _ = self.env.is_early_episode_termination(cost=cost[0], time_steps=time_steps, total_cost=sum(c))
                    done = done or early_done
                    self.buffer.append(action, x_prime, cost[0], done)
                    
                    if verbose: print x,action,x_prime,cost
                    
                    c.append(cost[0].tolist())
                    g.append(cost[1:].tolist())

                    x = x_prime
                trial_c.append(c)
                trial_g.append(g)

            all_c.append(np.mean([self.discounted_sum(x, self.gamma) for x in trial_c]))
            all_g.append(np.mean([ [self.discounted_sum(cost, self.gamma) for cost in np.array(x).T] for x in trial_g], axis=0).tolist())
            # all_g.append(np.mean([self.discounted_sum(x, self.gamma) for x in trial_g]))
        
        c = np.mean(all_c, axis=0)
        g = np.mean(all_g, axis=0)

        return c,g


    def determinstic_env_and_greedy_policy(self, policy, render=False, verbose=False, to_monitor=False, **kw):
        '''
        Run the evaluator
        '''

        all_c = []
        all_g = []
        for pi in policy:
            c = []
            g = []
            self.buffer = Buffer(num_frame_stack= self.num_frame_stack,
                                     buffer_size= self.buffer_size,
                                     min_buffer_size_to_train= self.min_buffer_size_to_train,
                                     pic_size = self.pic_size,)
            x = self.env.reset()
            self.buffer.start_new_episode(x)
            done = False
            time_steps = 0
            if to_monitor:
                monitor = Monitor(self.env, 'videos')
            while not done:
                if (self.env.env_type in ['car']) or render: 
                    if to_monitor: monitor.save()
                    # self.env.render()
                time_steps += 1
                
                action = pi(self.buffer.current_state())[0]
                print self.action_space_map[action]
                # import pdb; pdb.set_trace()
                cost = []
                for _ in range(self.frame_skip):
                    x_prime, costs, done, _ = self.env.step(self.action_space_map[action])
                    # if self.render:
                    self.env.render()
                    cost.append(costs)
                    if done:
                        break
                
                cost = np.vstack([np.hstack(x) for x in cost]).sum(axis=0)
                if self.constraint_thresholds is not None: 
                    cost[1:][self.constraints_cared_about] = np.array(cost[1:])[self.constraints_cared_about] >= self.constraint_thresholds[:-1]
                
                
                # early_done, _ = self.env.is_early_episode_termination(cost=cost[0], time_steps=time_steps, total_cost=sum(c))
                done = done #or early_done

                self.buffer.append(action, x_prime, cost[0], done)
                
                if verbose: print x,action,x_prime,cost
                #print time_steps, cost[0], action
                if (time_steps % 50) ==0 : print time_steps, cost[0], action
                c.append(cost[0])
                g.append(cost[1:])

                # x_prime , cost, done, _ = self.env.step(self.action_space_map[action])
                # done = done or self.env.is_early_episode_termination(cost=cost[0], time_steps=time_steps)
                # self.buffer.append(action, x_prime, cost[0], done)
                
                # if verbose: print x,action,x_prime,cost
                # if render: self.env.render()
                # c.append(cost[0])
                # g.append(cost[1])

                x = x_prime
            all_c.append(c)
            all_g.append(g)

            if to_monitor: monitor.make_video()
            if self.env.env_type in ['car']:  
                print 'Performance: %s/%s = %s' %  (self.env.tile_visited_count, len(self.env.track), self.env.tile_visited_count/float(len(self.env.track)))
        c = np.mean([self.discounted_sum(x, self.gamma) for x in all_c])
        g = np.mean([ [self.discounted_sum(cost, self.gamma) for cost in np.array(x).T] for x in all_g], axis=0).tolist()
        # g = np.mean([self.discounted_sum(np.array(x), self.gamma) for x in all_g], axis=0).tolist()

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

class Monitor(object):
    def __init__(self, env, filepath):
        self.frame_num = 0
        self.vid_num = 0
        self.filepath = os.path.join(os.getcwd(), filepath)
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)
        self.image_name = "image%05d.png"
        self.env = env
        self.images = []

    def save(self):
        import matplotlib.pyplot as plt
        full_path = os.path.join(self.filepath, self.image_name % self.frame_num)
        self.images.append(full_path)
        plt.imsave(full_path, self.env.render('rgb_array'))
        self.frame_num += 1

    def make_video(self):
        import subprocess
        current_dir = os.getcwd()
        os.chdir(self.filepath)
        subprocess.call([
            'ffmpeg', '-framerate', '8', '-i', self.image_name, '-r', '30', '-pix_fmt', 'yuv420p',
            'car_vid_%s.mp4' % self.vid_num
        ])

        self.vid_num += 1
        for file_name in self.images:
            os.remove(file_name)

        os.chdir(current_dir)

        


