import keras
import numpy as np
from replay_buffer import Buffer
import time
from keras.callbacks import ModelCheckpoint
import os

class DeepQLearning(object):
    def __init__(self, env, 
                       gamma, 
                       model_type='mlp', 
                       action_space_map = None,
                       num_iterations = 5000, 
                       sample_every_N_transitions = 10,
                       batchsize = 1000,
                       copy_over_target_every_M_training_iterations = 100,
                       max_time_spent_in_episode = 100,
                       buffer_size = 10000,
                       num_frame_stack=1,
                       min_buffer_size_to_train=1000,
                       frame_skip = 1,
                       pic_size = (96, 96),
                       models_path = None,
                       ):

        self.models_path = models_path
        self.env = env
        self.num_iterations = num_iterations
        self.gamma = gamma
        self.frame_skip = frame_skip
        _ = self.env.reset()
        if self.env.env_type in ['car']: self.env.render()
        _, r, _, _ = self.env.step(action_space_map[0])
        self.buffer = Buffer(buffer_size=buffer_size, num_frame_stack=num_frame_stack, min_buffer_size_to_train=min_buffer_size_to_train, pic_size = pic_size, n_costs = (len(np.hstack(r)),))        
        self.sample_every_N_transitions = sample_every_N_transitions
        self.batchsize = batchsize
        self.copy_over_target_every_M_training_iterations = copy_over_target_every_M_training_iterations
        self.max_time_spent_in_episode = max_time_spent_in_episode
        self.action_space_map = action_space_map

    def min_over_a(self, *args, **kw):
        return self.Q.min_over_a(*args, **kw)

    def all_actions(self, *args, **kw):
        return self.Q.all_actions(*args, **kw)

    # def representation(self, *args, **kw):
    #     return self.Q.representation(*args, **kw)

    def learn(self):
        
        more_callbacks = [ModelCheckpointExtended(self.models_path)]
        self.time_steps = 0
        training_iteration = -1
        perf = Performance()
        main_tic = time.time()
        for i in range(self.num_iterations):
            tic = time.time()
            x = self.env.reset()
            self.buffer.start_new_episode(x)
            done = False
            time_spent_in_episode = 0
            episode_cost = 0
            while not done:
                # if self.env.env_type in ['car']: self.env.render()
                
                time_spent_in_episode += 1
                self.time_steps += 1
                # print time_spent_in_episode
                
                use_random = np.random.rand(1) < self.epsilon(epoch=i, total_steps=self.time_steps)
                if use_random:
                    action = self.sample_random_action()
                else:
                    action = self.Q(self.buffer.current_state())[0]

                if (i % 50) == 0: print use_random, action, self.Q(self.buffer.current_state())[0], self.Q.all_actions(self.buffer.current_state())

                # import pdb; pdb.set_trace()
                # state = self.buffer.current_state()
                # import matplotlib.pyplot as plt
                # plt.imshow(state[-1])
                # plt.show()
                # self.Q.all_actions(state)

                cost = []
                for _ in range(self.frame_skip):
                    if done: continue
                    x_prime, costs, done, _ = self.env.step(self.action_space_map[action])
                    import pdb; pdb.set_trace()
                    cost.append(costs)

                cost = np.vstack([np.hstack(x) for x in cost]).sum(axis=0)
                early_done, punishment = self.env.is_early_episode_termination(cost=cost[0], time_steps=time_spent_in_episode, total_cost=episode_cost)
          
                if early_done:
                    cost[0] = cost[0] +  punishment
                done = done or early_done
                
                # self.buffer.append([x,action,x_prime, cost[0], done])
                self.buffer.append(action, x_prime, cost, done)

                # train
                is_train = ((self.time_steps % self.sample_every_N_transitions) == 0) and self.buffer.is_enough()

                if is_train:
                    # for _ in range(len(self.buffer.data)/self.sample_every_N_transitions):
                    training_iteration += 1
                    if (training_iteration % self.copy_over_target_every_M_training_iterations) == 0: 
                        self.Q.copy_over_to(self.Q_target)
                    batch_x, batch_a, batch_x_prime, batch_cost, batch_done = self.buffer.sample(self.batchsize)

                    target = batch_cost[:,0] + self.gamma*self.Q_target.min_over_a(np.stack(batch_x_prime))[0]*(1-batch_done)
                    X = [batch_x, batch_a]
                    
                    evaluation = self.Q.fit(X,target,epochs=1, batch_size=32, evaluate=False,verbose=False,tqdm_verbose=False, additional_callbacks=more_callbacks)
                
                x = x_prime

                episode_cost += cost[0]

            if self.env.env_type == 'car': 
                perf.append(float(self.env.tile_visited_count)/len(self.env.track))
            else:
                perf.append(episode_cost/self.env.min_cost)

            if (i % 1) == 0:
                print 'Episode %s' % i
                episode_time = time.time()-tic
                print 'Total Time: %s. Episode time: %s. Time/Frame: %s' % (np.round(time.time() - main_tic,2), np.round(episode_time, 2), np.round(episode_time/time_spent_in_episode, 2))
                print 'Episode frames: %s. Total frames: %s. Total train steps: %s' % (time_spent_in_episode, self.time_steps, training_iteration)
                if self.env.env_type in ['car']:
                    print 'Performance: %s/%s. Score out of 1: %s. Average Score: %s' %  (self.env.tile_visited_count, len(self.env.track), perf.last(), perf.get_avg_performance())
                else:
                    print 'Score out of 1: %s. Average Score: %s' %  (perf.last(), perf.get_avg_performance())
                print '*'*20
            if perf.reached_goal():
                return more_callbacks[0].all_filepaths[-1]
                training_complete = True#return self.Q #more_callbacks[0].all_filepaths[-1]
        self.buffer.save(os.path.join(os.getcwd(),'%s_data_{0}.h5' % self.env.env_type))

    def __call__(self,*args):
        return self.Q.__call__(*args)

    def __deepcopy__(self, memo):
        return self

class Performance(object):
    def __init__(self):
        self.goal = .85
        self.avg_over = 100
        self.costs = []

    def reached_goal(self):
        if self.get_avg_performance() >= self.goal:
            return True
        else:
            return False

    def append(self, cost):
        self.costs.append(cost)

    def last(self):
        return np.round(self.costs[-1], 3)

    def get_avg_performance(self):
        num_iters = min(self.avg_over, len(self.costs))
        return np.round(sum(self.costs[-num_iters:])/ float(num_iters), 3)


class ModelCheckpointExtended(ModelCheckpoint):
    def __init__(self, filepath, max_to_keep=5, monitor='loss', *args, **kw):
        super(ModelCheckpointExtended, self).__init__(filepath, *args, **kw)
        self.max_to_keep = max_to_keep
        self.all_filepaths = []

    def on_epoch_end(self, epoch, logs=None):
        
        super(ModelCheckpointExtended, self).on_epoch_end(epoch, logs)
        logs = logs or {}
        filepath = self.filepath.format(epoch=epoch + 1, **logs)
        
        self.all_filepaths.append(filepath)
        if len(self.all_filepaths) > self.max_to_keep:
            try:
                os.remove(self.all_filepaths.pop(0))
            except:
                pass


# class Buffer(object):
#     def __init__(self, buffer_size=10000):
#         self.data = []
#         self.size = buffer_size
#         self.idx = -1

#     def append(self, datum):
#         self.idx = (self.idx + 1) % self.size
        
#         if len(self.data) > self.idx:
#             self.data[self.idx] = datum
#         else:
#             self.data.append(datum)

#     def sample(self, N):
#         N = min(N, len(self.data))
#         rows = np.random.choice(len(self.data), size=N, replace=False)
#         return np.array(self.data)[rows]



