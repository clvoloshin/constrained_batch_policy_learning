import h5py
import numpy as np
import deepdish as dd
#from thread_safe import threadsafe_generator
import threading

import keras
from keras.models import Sequential, Model, load_model, model_from_config
from keras.layers import Dense, Conv2D, Flatten, Input, concatenate, Lambda, MaxPooling2D, Dropout, dot
from keras import optimizers
from keras import initializers
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import Callback, TensorBoard
from keras.backend import eval

from car_racing import ExtendedCarRacing
import itertools
from exact_policy_evaluation import ExactPolicyEvaluator

from pyvirtualdisplay import Display
display = Display(visible=0, size=(1280, 1024))
display.start()

# env = gym.make('CarRacing-v0')
constraint_thresholds = [1., 15.] + [1]
constraints_cared_about = [-1,2]
constraints = [300*.1, 300*.1] + [0,0,0,0,0]
pic_size = (96, 96,3)
num_frame_stack=3
frame_skip=3
gamma=.95
action_space_map = {}
for i, action in enumerate([k for k in itertools.product([-1, 0, 1], [1, 0], [0.2, 0])]):
    action_space_map[i] = action

init_seed = 2
stochastic_env = False # = not deterministic
max_pos_costs = 12 # The maximum allowable positive cost before ending episode early
max_time_spent_in_episode = 2000
env = ExtendedCarRacing(init_seed, stochastic_env, max_pos_costs)
exact_policy_algorithm = ExactPolicyEvaluator(action_space_map, gamma, env=env, frame_skip=frame_skip, num_frame_stack=num_frame_stack, pic_size = pic_size, constraint_thresholds=constraint_thresholds, constraints_cared_about=constraints_cared_about)
env.reset()


GPU = 0 
SEED = 0
np.random.seed(SEED)
import tensorflow as tf
tf.set_random_seed(SEED)
import random
random.seed(SEED)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.allow_growth = True
from keras import backend as K
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

LEARNING_RATE = 0.0005
dim_of_actions = 12
input_shape = (96,96,3)
gamma = 0.95

class threadsafe_iter:
	"""Takes an iterator/generator and makes it thread-safe by
	serializing call to the `next` method of given iterator/generator.
	"""
	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def next(self):
		with self.lock:
			return self.it.next()


def threadsafe_generator(f):
	"""A decorator that takes a generator function and makes it thread-safe.
	"""
	def g(*a, **kw):
		return threadsafe_iter(f(*a, **kw))
	return g

class NN:
	def __init__(self, gpu=0):
		self.gpu = gpu
		rmsProp = optimizers.RMSprop(lr=LEARNING_RATE, rho=0.95, epsilon=1e-08, decay=0.0)

		def init(): return keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=np.random.randint(2**32))
		with tf.device('/gpu:'+str(self.gpu)):
			model = Sequential()
			model.add(Conv2D(8, (7,7), strides = 3, activation = 'relu', padding = 'same', input_shape = (96,96,3),kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6)))
			model.add(MaxPooling2D())
			#model.add(Dropout(0.25))
			model.add(Conv2D(16,(3,3), strides = 1, activation = 'relu', padding = 'same',kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6)))
			model.add(MaxPooling2D())
			#model.add(Dropout(0.25))
			model.add(Flatten())
			model.add(Dense(256, activation='relu',kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6)))
			#model.add(Dropout(0.5))
			model.add(Dense(dim_of_actions, name='all_actions', activation="linear",kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6)))

			self.model = model
			self.compile()
			self.model._make_predict_function()
		#self.model.summary()
		
	def compile(self):
		def huber_loss(y_true, y_pred, clip_value):
			# Huber loss, see https://en.wikipedia.org/wiki/Huber_loss and
			# https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
			# for details.
			assert clip_value > 0.

			x = y_true - y_pred
			if np.isinf(clip_value):
				# Spacial case for infinity since Tensorflow does have problems
				# if we compare `K.abs(x) < np.inf`.
				return .5 * K.square(x)

			condition = K.abs(x) < clip_value
			squared_loss = .5 * K.square(x)
			linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
			if K.backend() == 'tensorflow':
				import tensorflow as tf
				if hasattr(tf, 'select'):
					return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
				else:
					return tf.where(condition, squared_loss, linear_loss)  # condition, true, false
			elif K.backend() == 'theano':
				from theano import tensor as T
				return T.switch(condition, squared_loss, linear_loss)
			else:
				raise RuntimeError('Unknown backend "{}".'.format(K.backend()))

		def mean_pred(y_true, y_pred):
			return K.mean(y_pred)

		def min_pred(y_true, y_pred):
			return K.min(y_pred)

		def clipped_masked_error(args):
				y_true, y_pred, mask = args
				loss = huber_loss(y_true, y_pred, 10)
				loss *= mask  # apply element-wise mask
				return K.sum(loss, axis=-1)
		# Create trainable model. The problem is that we need to mask the output since we only
		# ever want to update the Q values for a certain action. The way we achieve this is by
		# using a custom Lambda layer that computes the loss. This gives us the necessary flexibility
		# to mask out certain parameters by passing in multiple inputs to the Lambda layer.
		y_pred = self.model.output
		y_true = Input(name='y_true', shape=(dim_of_actions,))
		mask = Input(name='mask', shape=(dim_of_actions,))
		loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='huber')([y_pred, y_true, mask])
		#predicted_value = Lambda(value_pred, output_shape=(1,), name='predicted_value')([y_pred, mask])
		#ins = [self.model.input] if type(self.model.input) is not list else self.model.input
		ins = self.model.input
		#trainable_model = Model(inputs=ins + [y_true, mask], outputs=[loss_out, y_pred])
		trainable_model = Model(inputs=[ins,y_true, mask], outputs=[loss_out, y_pred])
		assert len(trainable_model.output_names) == 2
		#combined_metrics = {trainable_model.output_names[1]: metrics}
		losses = [
			lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
			lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
		]
		#trainable_model.compile(optimizer=optimizer, loss=losses, metrics=combined_metrics)
		rmsProp = optimizers.RMSprop(lr=LEARNING_RATE, rho=0.95, epsilon=1e-08, decay=0.0)
		#opt = optimizers.Adam(lr=0.0001, clipnorm = 10)
		#trainable_model.compile(optimizer=rmsProp, loss=losses)
		trainable_model.compile(optimizer=rmsProp, loss=losses, metrics = [min_pred])
		#trainable_model.compile(optimizer='adam', loss=losses, metrics = [min_pred])
		self.trainable_model = trainable_model
		#print self.trainable_model.summary()
		#print self.trainable_model.metrics_names
		#time.sleep(5)

		self.compiled = True

	def saveWeight(self):
		self.model.save_weights('fqi_model.h5')

	def loadWeight(self):
		#path = 'weight/'
		self.model.load_weights('fqi_model.h5')
		self.model.reset_states()

	def clear_memory(self):
		del self.model


@threadsafe_generator
def data_generator(indices, fixed_permutation=False, batch_size = 64):
	#data_length = len(dataset['done']) - 1 ## Maybe throw out the very last data point to avoid out of range index error
	data_length = len(indices)
	number_of_batches = int(np.floor(data_length/float(batch_size)))
	#random_permutation = np.random.permutation(np.arange(data_length))
	random_permutation = np.random.permutation(indices)
	i= -1
	while True:
		i = (i+1) % number_of_batches
		idxs = random_permutation[(i*batch_size):((i+1)*batch_size)]

		#print idxs
		x = np.rollaxis(dataset['frames'][dataset['prev_states'][idxs]],1,4)
		a = dataset['a'][idxs] ## need to make it 2d?
		x_prime = np.rollaxis(dataset['frames'][dataset['next_states'][idxs]],1,4)
		c = dataset['c'][idxs] ## scaling the cost back?
		g = dataset['g'][idxs]
		dones = dataset['done'][idxs]

		target_q_values = Q_k_minus_1.model.predict(x_prime)
		assert target_q_values.shape == (batch_size, dim_of_actions)
		q_batch = np.min(target_q_values, axis=1) ## we're minimizing cost
		assert q_batch.shape == (batch_size,)
		
		targets = np.zeros((batch_size, dim_of_actions))
		dummy_targets = np.zeros((batch_size,))
		masks = np.zeros((batch_size, dim_of_actions))

		discounted_q_batch = gamma * q_batch
		terminalBatch = np.array([1-float(done) for done in dones])
		assert terminalBatch.shape == (batch_size,)
		discounted_q_batch *= terminalBatch
		assert c.shape == discounted_q_batch.shape
		cost_to_go_batch = c + discounted_q_batch
		
		for idx, (target, mask, value, action) in enumerate(zip(targets, masks, cost_to_go_batch, a)):
			target[action] = value  # update action with estimated accumulated reward
			dummy_targets[idx] = value
			mask[action] = 1.  # enable loss for this specific action

		assert x.shape == (batch_size, 96,96,3)
		assert targets.shape == (batch_size, 12)
		#assert sum(masks) == batch_size

		yield ([x, targets, masks], [dummy_targets, targets])

@threadsafe_generator
def validation_generator(indices, fixed_permutation=False, batch_size = 64):
	#data_length = len(dataset['done']) - 1 ## Maybe throw out the very last data point to avoid out of range index error
	data_length = len(indices)
	number_of_batches = int(np.floor(data_length/float(batch_size)))
	#random_permutation = np.random.permutation(np.arange(data_length))
	random_permutation = np.random.permutation(indices)
	i= -1
	while True:
		i = (i+1) % number_of_batches
		idxs = random_permutation[(i*batch_size):((i+1)*batch_size)]

		#print idxs
		x = np.rollaxis(dataset['frames'][dataset['prev_states'][idxs]],1,4)
		a = dataset['a'][idxs] ## need to make it 2d?
		x_prime = np.rollaxis(dataset['frames'][dataset['next_states'][idxs]],1,4)
		c = dataset['c'][idxs]## scaling the cost back?
		g = dataset['g'][idxs]
		dones = dataset['done'][idxs]

		target_q_values = Q_k_minus_1.model.predict(x_prime)
		assert target_q_values.shape == (batch_size, dim_of_actions)
		q_batch = np.min(target_q_values, axis=1) ## we're minimizing cost
		assert q_batch.shape == (batch_size,)
		
		targets = np.zeros((batch_size, dim_of_actions))
		dummy_targets = np.zeros((batch_size,))
		masks = np.zeros((batch_size, dim_of_actions))

		discounted_q_batch = gamma * q_batch
		terminalBatch = np.array([1-float(done) for done in dones])
		assert terminalBatch.shape == (batch_size,)
		discounted_q_batch *= terminalBatch
		assert c.shape == discounted_q_batch.shape
		cost_to_go_batch = c + discounted_q_batch
		
		for idx, (target, mask, value, action) in enumerate(zip(targets, masks, cost_to_go_batch, a)):
			target[action] = value  # update action with estimated accumulated reward
			dummy_targets[idx] = value
			mask[action] = 1.  # enable loss for this specific action

		assert x.shape == (batch_size, 96,96,3)
		assert targets.shape == (batch_size, 12)
		#assert sum(masks) == batch_size

		yield ([x, targets, masks], [dummy_targets, targets])

def clone_model(model, custom_objects={}):
	# Requires Keras 1.0.7 since get_config has breaking changes.
	config = {
		'class_name': model.__class__.__name__,
		'config': model.get_config(),
	}
	clone = model_from_config(config, custom_objects=custom_objects)
	clone._make_predict_function()
	clone.set_weights(model.get_weights())
	return clone

def weight_change_norm(model, target_model):
	norm_list = []
	number_of_layers = len(model.layers)
	for i in range(number_of_layers):
		model_matrix = model.layers[i].get_weights()
		target_model_matrix = target_model.layers[i].get_weights()
		if len(model_matrix) >0:
			#print "layer ", i, " has shape ", model_matrix[0].shape
			if model_matrix[0].shape[0] > 0:
				norm_change = np.linalg.norm(model_matrix[0]-target_model_matrix[0])
				norm_list.append(norm_change)
	return sum(norm_list)*1.0/len(norm_list)


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


action_data = dd.io.load('./seed_2/car_data_actions_seed_2.h5')
frame_data = dd.io.load('./seed_2/car_data_frames_seed_2.h5')
done_data = dd.io.load('./seed_2/car_data_is_done_seed_2.h5')
next_state_data = dd.io.load('./seed_2/car_data_next_states_seed_2.h5')
current_state_data = dd.io.load('./seed_2/car_data_prev_states_seed_2.h5')
cost_data = dd.io.load('./seed_2/car_data_rewards_seed_2.h5')

frame_gray_scale = np.zeros((len(frame_data),96,96)).astype('float32')
for i in range(len(frame_data)):
	frame_gray_scale[i,:,:] = np.dot(frame_data[i,:,:,:]/255. , [0.299, 0.587, 0.114])

dataset = {'frames':frame_gray_scale,
			'prev_states': current_state_data,
			'next_states': next_state_data,
			'a': action_data,
			'c':cost_data[:,0]/20.3, ## Divide by the largest one
			'g':cost_data[:,1:],
			'done': done_data
			}


### Load data set
#dataset = dd.io.load('car_racing_data.h5')
data_length = len(frame_data)-1
### Start training


Q_k_minus_1 = NN(gpu = GPU) ## This is the target network, initialize it with something
Q_k = NN(gpu=GPU) ### Initialize the value network with something
#Q_k_minus_1.loadWeight() ### cheat: loading in DQN weights
## Form the data set?

number_of_iter = 100
batch_size = 32
epochs_per_iter = 1 ## per_iter
#steps_per_epoch = data_length / batch_size

#mcp_save = ModelCheckpoint('fqi_test_model.hdf5', save_best_only=False, mode='auto', period=1)

#reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
history = LossHistory()
iteration_losses = []
stop_training = False
lr_counter = 0
train_iter = 0
#while not stop_training:
for iteration in range(number_of_iter):
	print "------------"
	print "Iteration: ", train_iter
	lr = eval(Q_k.trainable_model.optimizer.lr)
	print "Current learning rate: ", lr
	lr_counter += 1
	## training validation split
	indices = np.random.permutation(np.arange(data_length))
	cutoff = int(1*data_length)
	train_idx = indices[:cutoff]
	valid_idx = indices[cutoff:]
	steps_per_epoch = len(train_idx) / batch_size
	valid_steps = len(valid_idx) / batch_size
	#gen = data_generator(dataset, fixed_permutation=False, batch_size=batch_size)
	gen = data_generator(train_idx, fixed_permutation=False, batch_size=batch_size)
	#valid_gen = validation_generator(valid_idx, fixed_permutation=False, batch_size=batch_size)

	#mcp_save = ModelCheckpoint('FQI_models/fqi_model_1epoch_gamma095_lr00025_'+str(iteration)+'.hdf5', save_best_only=True, monitor='val_loss', mode='min')
	#Q_k.trainable_model.fit_generator(gen, epochs=epochs_per_iter, steps_per_epoch=steps_per_epoch, max_queue_size=10, workers=8, use_multiprocessing=False, verbose=1, validation_data = valid_gen, validation_steps = valid_steps, callbacks=[history])
	Q_k.trainable_model.fit_generator(gen, epochs=epochs_per_iter, steps_per_epoch=steps_per_epoch, max_queue_size=10, workers=8, use_multiprocessing=False, verbose=1, callbacks=[history])
	iter_loss = sum(history.losses) *1.0/ len(history.losses)
	#print "This iteration loss: ", iter_loss
	iteration_losses.append(iter_loss)
	"""
	if len(iteration_losses) > 5 and iteration_losses[-1]>max(iteration_losses[-6:-1]) and lr_counter >=5:
		if lr > 0.0001:
			lr = max(0.0001, lr*0.5)
			K.set_value(Q_k.trainable_model.optimizer.lr,lr)
			lr_counter = 0
		else:
			stop_training = True
	"""
	#Q_k.trainable_model.fit_generator(gen, epochs=epochs_per_iter, steps_per_epoch=steps_per_epoch, max_queue_size=10, workers=3, use_multiprocessing=False, verbose=0, validation_data = valid_gen, validation_steps = valid_steps)
	#Q_k_minus_1.model = clone_model(Q_k.model)
	## Test weight change in last layer
	old_matrix = Q_k_minus_1.model.layers[-1].get_weights()
	new_matrix = Q_k.model.layers[-1].get_weights()
	#print "dimension of weight layer ", new_matrix[0].shape
	#print "Norm of weight change is ", np.linalg.norm(new_matrix[0]-old_matrix[0])
	print "Norm of weight change is ", weight_change_norm(Q_k.model, Q_k_minus_1.model)
	print
	print exact_policy_algorithm.run(Q_k)	
	Q_k_minus_1.model.set_weights(Q_k.model.get_weights())
	Q_k.model.save('FQI_models/fqi_model_1epoch_gamma095_lr0005_fixed_'+str(train_iter)+'.hdf5')
	train_iter += 1

	#Q_k.compile() ## reset optimizer state
	#Q_k.model.reset_states()
	#Q_k.trainable_model.reset_states()


### Copying model of Q_k over to Q_k_minus_1 before repeating
