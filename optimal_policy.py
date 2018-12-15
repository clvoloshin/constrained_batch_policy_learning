from model import Model
import keras
import numpy as np

class DeepQLearning(object):
    def __init__(self, env, gamma):
        self.env = env
        self.state_space_dim = env.nS
        self.action_space_dim = env.nA
        self.Q = Model(self.state_space_dim+self.action_space_dim, 1, self.action_space_dim, gamma)
        self.Q_target = Model(self.state_space_dim+self.action_space_dim, 1, self.action_space_dim, gamma)
        self.num_iterations = 5000
        self.gamma = gamma
        self.buffer = Buffer()
        self.sample_every_N_transitions = 10
        self.batchsize = 1000
        self.copy_over_target_every_M_training_iterations = 100


        self.learn()

    def learn(self):
        
        time_steps = 0
        training_iteration = -1
        costs = []
        for i in range(self.num_iterations):
            x = self.env.reset()
            done = False
            time_spent_in_episode = 0
            episode_cost = 0
            while not done:
                time_spent_in_episode += 1
                time_steps += 1
                
                action = self.Q(np.eye(1, self.state_space_dim, x))[0]
                if np.random.rand(1) < self.epsilon(i):
                    action = np.random.choice(self.action_space_dim)
                
                x_prime , reward, done, _ = self.env.step(action)

                self.buffer.append([x,action,x_prime,-reward])

                # train
                if (time_steps % self.sample_every_N_transitions) == 0:
                    # for _ in range(len(self.buffer.data)/self.sample_every_N_transitions):
                    training_iteration += 1
                    if (training_iteration % self.copy_over_target_every_M_training_iterations) == 0: 
                        self.copy_over(self.Q, self.Q_target)
                    batch = self.buffer.sample(self.batchsize)

                    target = batch[:,3] + self.gamma*self.Q_target.min_over_a(np.eye(self.state_space_dim)[batch[:,2].astype(int)])[0]
                    X = np.hstack([np.eye(self.state_space_dim)[batch[:,0].astype(int)], np.eye(self.action_space_dim)[batch[:,1].astype(int)]])
                    
                    evaluation = self.Q.fit(X,target,epochs=1, batch_size=32,verbose=False)
                
                x = x_prime

                episode_cost -= reward

            costs.append(episode_cost)

            if (i % 50) == 0:
                print 'Iteration %s performance: %s' % (i, np.abs(np.mean(costs[-200:])))
            if np.abs(np.mean(costs[-200:])) >= .95:
                return

    def copy_over(self, from_, to_):
        to_.model = keras.models.clone_model(from_.model)
        to_.model.set_weights(from_.model.get_weights())

    def epsilon(self, iteration):
        return 1./(iteration/100 + 3)

    def __call__(self,*args):
        self.Q.__call__(*args)

class Buffer(object):
    def __init__(self):
        self.data = []
        self.size = 10000
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






