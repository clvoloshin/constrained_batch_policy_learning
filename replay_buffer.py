
import numpy as np
import deepdish as dd

class Buffer(object):
    """
    This saves the agent's experience in windowed cache.
    Each frame is saved only once but state is stack of num_frame_stack frames

    In the beginning of an episode the frame-stack is padded
    with the beginning frame
    """

    def __init__(self,
            num_frame_stack=1,
            buffer_size=10000,
            min_buffer_size_to_train=1000,
            pic_size = (96,96),
            action_space_dim = 4,
            n_costs = (),
    ):
        self.n_costs = n_costs
        self.pic_size = pic_size
        self.action_space_dim = action_space_dim
        self.num_frame_stack = num_frame_stack
        self.capacity = buffer_size
        self.counter = 0
        self.frame_window = None
        self.max_frame_cache = self.capacity + 2 * self.num_frame_stack + 1
        self.init_caches()
        self.expecting_new_episode = True
        self.min_buffer_size_to_train = min_buffer_size_to_train
        self.data = {'x':[], 'a':[], 'x_prime':[], 'c':[], 'g':[], 'done':[], 'cost':[]}

    def append(self, action, frame, reward, done):
        assert self.frame_window is not None, "start episode first"
        self.counter += 1
        frame_idx = self.counter % self.max_frame_cache
        exp_idx = (self.counter - 1) % self.capacity

        self.prev_states[exp_idx] = self.frame_window
        self.frame_window = np.append(self.frame_window[1:], frame_idx)
        self.next_states[exp_idx] = self.frame_window
        self.actions[exp_idx] = action
        self.is_done[exp_idx] = done
        self.frames[frame_idx] = frame
        self.rewards[exp_idx] = reward
        if done:
            self.expecting_new_episode = True

    def start_new_episode(self, frame):
        # it should be okay not to increment counter here
        # because episode ending frames are not used
        assert self.expecting_new_episode, "previous episode didn't end yet"
        frame_idx = self.counter % self.max_frame_cache
        self.frame_window = np.repeat(frame_idx, self.num_frame_stack)
        self.frames[frame_idx] = frame
        self.expecting_new_episode = False

    def is_over(self):
        return self.expecting_new_episode

    def get_length(self):
        return min(self.capacity, self.counter)

    def sample(self, N):
        count = min(self.capacity, self.counter)
        minimum = max(count-40000, 0) # UNHARDCODE THIS. THIS IS FOR USING BUFFER AS SAVER + Exp Replay
        batchidx = np.random.randint(minimum, count, size=N)

        x = self.frames[self.prev_states[batchidx]]
        action = self.actions[batchidx]
        x_prime = self.frames[self.next_states[batchidx]]
        reward = self.rewards[batchidx]
        done = self.is_done[batchidx]
        
        return [x, action, x_prime, reward, done]

    def get_all(self, key):
        valid_states = min(self.capacity, self.counter)
        if key == 'x':
            return self.frames[self.prev_states[:valid_states]]
        elif key == 'a':
            return self.actions[:valid_states]
        elif key == 'x_prime':
            return self.frames[self.next_states[:valid_states]]
        elif key == 'c':
            return self.rewards[:valid_states][:, 0]
        elif key == 'g':
            return self.rewards[:valid_states][:, 1:]
        elif key == 'done':
            return self.is_done[:valid_states]
        elif key == 'cost':
            return []
        elif key == 'frames':
            maximum = max(np.max(self.prev_states[:valid_states]), np.max(self.next_states[:valid_states])) + 1
            return self.frames[:maximum]
        elif key == 'prev_states':
            return self.prev_states[:valid_states]
        elif key == 'next_states':
            return self.next_states[:valid_states]
        else:
            raise
            
    def is_enough(self):
        return self.counter > self.min_buffer_size_to_train

    def current_state(self):
        # assert not self.expecting_new_episode, "start new episode first"'
        assert self.frame_window is not None, "do something first"
        if len(self.pic_size) == 2:
            return np.rollaxis(self.frames[self.frame_window], 0,3)
        else:
            return self.frames[self.frame_window]

    def init_caches(self):
        self.rewards = np.empty((self.capacity,) + self.n_costs, dtype="float64")
        self.prev_states = np.empty((self.capacity, self.num_frame_stack), dtype="uint32")
        self.next_states = np.empty((self.capacity, self.num_frame_stack), dtype="uint32")
        self.is_done = np.empty(self.capacity, "uint8")
        self.actions = np.empty((self.capacity), dtype="uint8")
        self.frames = np.empty((self.max_frame_cache,) + self.pic_size, dtype="uint8")

    def get_state_action_pairs(self, env_type=None):
        if 'state_action' in self.data:
            return self.data['state_action']
        else:
            if env_type == 'lake':
                pairs = [np.array(self.data['x']), np.array(self.data['a']).reshape(1,-1).T ]
            elif env_type == 'car':
                pairs = [np.array(self.data['x']), np.array(self.data['a']).reshape(1,-1).T ]
            self.data['state_action'] = pairs

    def calculate_cost(self, lamb):
        self.scale = np.max(np.abs(np.array(self.data['c'] + np.dot(lamb[:-1], np.array(self.data['g']).T))))
        costs = np.array(self.data['c'] + np.dot(lamb[:-1], np.array(self.data['g']).T))/self.scale


        # costs = costs/np.max(np.abs(costs))
        self.data['cost'] = costs.tolist()

    def set_cost(self, key, idx=None):
        if key == 'g': assert idx is not None, 'Evaluation must be done per constraint until parallelized'

        if key == 'c':
            self.scale = np.max(np.abs(self.data['c']))
            self.data['cost'] = self.data['c']/self.scale
        elif key == 'g':
            # Pick the idx'th constraint
            self.scale = np.max(np.abs(self.data['g'][:,idx]))
            self.data['cost'] = self.data['g'][:,idx]/self.scale
        else:
            raise

    def preprocess(self, env_type):

        for key in self.data:
            self.data[key] = self.get_all(key)

    def save(self, path):
        #data = {'frames':self.frames, 'prev_states':self.prev_states, 'next_states':self.next_states, 'rewards':self.rewards, 'is_done':self.is_done, 'actions':self.actions}
        #for data, key in zip([self.frames, self.prev_states, self.next_states, self.rewards, self.is_done, self.actions],['frames', 'prev_astates', 'next_states', 'costs', 'is_done', 'actions'])
        #       dd.io.save(path % key, data)
        count = min(self.capacity, self.counter)
        dd.io.save(path.format('frames'), self.frames[:count])
        dd.io.save(path.format('prev_states'), self.prev_states[:count])
        dd.io.save(path.format('next_states'), self.next_states[:count])
        dd.io.save(path.format('rewards'), self.rewards[:count])
        dd.io.save(path.format('is_done'), self.is_done[:count])
        dd.io.save(path.format('actions'), self.actions[:count])



class Dataset(Buffer):
    def __init__(self, num_frame_stack, pic_size, n_costs):
        
        self.pic_size = pic_size
        self.num_frame_stack = num_frame_stack
        self.data = {'frames':[], 'prev_states':[], 'a':[], 'next_states':[], 'c':[], 'g':[], 'done':[], 'cost':[], 'x_prime_repr':[], 'x_repr':[]}
        self.max_trajectory_length = 0
        self.n_costs = n_costs
        self.episodes = [Buffer(num_frame_stack=self.num_frame_stack,buffer_size=int(200000),min_buffer_size_to_train=0,pic_size = self.pic_size, n_costs = self.n_costs)]

    def append(self, *args):
        self.episodes[-1].append(*args)

        # update max_trajectory_length
        if self.episodes[-1].get_length() > self.max_trajectory_length:
            self.max_trajectory_length = self.episodes[-1].get_length()

    def start_new_episode(self, *args):
        # self.episodes.append(Buffer(num_frame_stack=self.num_frame_stack,buffer_size=int(2000),min_buffer_size_to_train=0,pic_size = self.pic_size, n_costs = self.n_costs))
        self.episodes[-1].start_new_episode(args[0])

    def current_state(self):
        return self.episodes[-1].current_state()
        
    def get_max_trajectory_length(self):
        return self.max_trajectory_length
        
    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, item):
        self.data[key] = item

    def __len__(self):
        return len(self.data['a'])-5

    def preprocess(self, env_type):

        for key in ['frames', 'prev_states', 'next_states', 'a', 'done', 'c', 'g']:
            self.data[key] = self.episodes[-1].get_all(key)
        
        # [x.preprocess(env_type) for x in self.episodes]

        # for key in self.data:
        #     if key in ['g', 'prev_states', 'next_states', 'frames']:
        #         try:
        #             self.data[key] = np.vstack([x.get_all[key] for x in self.episodes])#.tolist()
        #         except:
        #             self.data[key] = np.hstack([x.get_all[key] for x in self.episodes])#.tolist()
        #     else:
        #         self.data[key] = np.hstack([x.get_all[key] for x in self.episodes])#.tolist()

        #     if env_type == 'lake':
        #         if key in ['g']:
        #             try:
        #                 self.data[key] = np.vstack([x[key] for x in self.episodes]).tolist()
        #             except:
        #                 self.data[key] = np.hstack([x[key] for x in self.episodes]).tolist()
        #         else:
        #             self.data[key] = np.hstack([x[key] for x in self.episodes]).tolist()
        #     elif env_type == 'car':
        #         if key in ['g', 'x', 'x_prime']:
        #             try:
        #                 self.data[key] = np.vstack([x[key] for x in self.episodes]).tolist()
        #             except:
        #                 self.data[key] = np.hstack([x[key] for x in self.episodes]).tolist()
        #         else:
        #             self.data[key] = np.hstack([x[key] for x in self.episodes]).tolist()
        #     else:
        #         raise
        # [x.get_state_action_pairs(env_type) for x in self.episodes]
        # self.get_state_action_pairs(env_type)

    def get_state_action_pairs(self, env_type=None):
        # if 'state_action' in self.data:
        #     return self.data['state_action']
        # else:
        if env_type == 'lake':
            pairs = [np.array(self('x_repr')), np.array(self.data['a']).reshape(1,-1).T ]
        elif env_type == 'car':
            pairs = [np.array(self('x_repr')), np.array(self.data['a']).reshape(1,-1).T ]
        return pairs

    def calculate_cost(self, lamb):
        scale = max(np.abs(self.data['c'])) + np.dot(lamb, np.array([[1],[1],[0]]))
        costs = np.array(self.data['c'] + np.dot(lamb, np.array(self.data['g']).T))/scale
        
        # costs = costs/np.max(np.abs(costs))
        self.data['cost'] = costs

        # [x.calculate_cost(lamb) for x in self.episodes]

    def set_cost(self, key, idx=None):
        if key == 'g': assert idx is not None, 'Evaluation must be done per constraint until parallelized'

        if key == 'c':
            self.scale = np.max(np.abs(self.data['c']))
            self.data['cost'] = self.data['c']/self.scale
            # [x.set_cost('c') for x in self.episodes]
        elif key == 'g':
            # Pick the idx'th constraint
            self.scale = np.max(np.abs(np.array(self.data['g'])[:,idx]))
            self.data['cost'] = np.array(self.data['g'])[:,idx]/self.scale
            # [x.set_cost('g', idx) for x in self.episodes]
        else:
            raise
