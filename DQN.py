import keras
import numpy as np

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
                       ):
        self.env = env
        self.num_iterations = num_iterations
        self.gamma = gamma
        self.buffer = Buffer(buffer_size=buffer_size)
        self.sample_every_N_transitions = sample_every_N_transitions
        self.batchsize = batchsize
        self.copy_over_target_every_M_training_iterations = copy_over_target_every_M_training_iterations
        self.max_time_spent_in_episode = max_time_spent_in_episode
        self.action_space_map = action_space_map

    def min_over_a(self, *args, **kw):
        return self.Q.min_over_a(*args, **kw)

    def all_actions(self, *args, **kw):
        return self.Q.all_actions(*args, **kw)

    def learn(self):
        
        self.time_steps = 0
        training_iteration = -1
        costs = []
        for i in range(self.num_iterations):
            x = self.env.reset()
            done = False
            time_spent_in_episode = 0
            episode_cost = 0
            while not done:
                # if (i % 10 == 0): self.env.render()
                time_spent_in_episode += 1
                self.time_steps += 1
                # print time_spent_in_episode
                
                action = self.Q([x])[0]
                if np.random.rand(1) < self.epsilon(i):
                    action = self.sample_random_action()
                
                x_prime , cost, done, _ = self.env.step(self.action_space_map[action])
                done = done or self.env.is_early_episode_termination(cost[0])
                
                self.buffer.append([x,action,x_prime, cost[0], done])

                # train
                if (self.time_steps % self.sample_every_N_transitions) == 0:
                    # for _ in range(len(self.buffer.data)/self.sample_every_N_transitions):
                    training_iteration += 1
                    if (training_iteration % self.copy_over_target_every_M_training_iterations) == 0: 
                        self.Q.copy_over_to(self.Q_target)
                    batch = self.buffer.sample(self.batchsize)

                    target = batch[:,3] + self.gamma*self.Q_target.min_over_a(np.stack(batch[:,2]))[0]*(1-batch[:,4])
                    X = [batch[:,0], batch[:,1]]
                    
                    evaluation = self.Q.fit(X,target,epochs=1, batch_size=32,evaluate=False,verbose=False,tqdm_verbose=False)
                
                x = x_prime

                episode_cost += cost[0]
            costs.append(episode_cost/self.env.min_cost)

            if (i % 1) == 0:
                print 'Number of frames seen: %s' % self.time_steps
                print 'Iteration %s performance: %s. Average performance: %s' % (i, costs[-1], np.sum(costs[-200:])/200.)
            if (np.sum(costs[-200:])/200.) >= .85:
                return

    def __call__(self,*args):
        return self.Q.__call__(*args)

class Buffer(object):
    def __init__(self, buffer_size=10000):
        self.data = []
        self.size = buffer_size
        self.idx = -1

    def append(self, datum):
        self.idx = (self.idx + 1) % self.size
        
        if len(self.data) > self.idx:
            self.data[self.idx] = datum
        else:
            self.data.append(datum)

    def sample(self, N):
        N = min(N, len(self.data))
        rows = np.random.choice(len(self.data), size=N, replace=False)
        return np.array(self.data)[rows]






