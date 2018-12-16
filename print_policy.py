import numpy as np

class PrintPolicy(object):
	def __init__(self, size=[4,4]):
		self.mapping = {0:'<', 1:'v', 2:'>', 3:'^'}
		self.size = size

	def pprint(self, *args):
		if len(args) == 1:
			pi = args[0]
			size = self.size[0]*self.size[1]
			if isinstance(pi,(list,)):
				if len(pi) > 0:
					actions_for_each_pi = np.hstack([np.eye(len(self.mapping))[p.min_over_a(np.eye(size))[1].reshape(1,-1).T] for p in pi])
					policy = np.hstack([np.array(range(size)).reshape(1,-1).T, np.argmax(actions_for_each_pi.mean(1), 1).reshape(1,-1).T])

					Qs_for_each_pi = np.vstack([np.array([p.all_actions(np.eye(size))]) for p in pi])
					Q = np.hstack([np.array(range(size)).reshape(1,-1).T, np.mean(Qs_for_each_pi,axis=0)[np.arange(Qs_for_each_pi.shape[1]),policy[:,1]].reshape(-1,1)])
				else:
					return
			else:
				policy = np.hstack([np.array(range(size)).reshape(1,-1).T, pi.min_over_a(np.eye(size))[1].reshape(1,-1).T])
				Q = np.hstack([np.array(range(size)).reshape(1,-1).T, pi.min_over_a(np.eye(size))[0].reshape(1,-1).T])
		elif len(args) == 2:
			X_a, costs = args
			idxs = np.unique(X_a, axis=0, return_index=True)[1]
			x_a_c = np.hstack([np.argmax(X_a[idxs][:,:-4],1).reshape(1,-1).T, np.argmax(X_a[idxs][:,-4:],1).reshape(1,-1).T, costs[idxs].reshape(1,-1).T])
			policy = np.array([[y[0][0], y[np.argmin(y[:,2],0),1]] for y in np.array_split(x_a_c,np.unique(x_a_c[:,0]).shape[0])])
			Q = np.array([[y[0][0], np.min(y,0)[1]] for y in np.array_split(x_a_c[:,[0,2]],np.unique(x_a_c[:,0]).shape[0])])
		else:
			raise

		direction_grid = [['H' for x in range(self.size[1])] for y in range(self.size[0])]
		direction_grid[-1][-1] = 'G'

		Q_grid = [['  H  ' for x in range(self.size[1])] for y in range(self.size[0])]
		Q_grid[-1][-1] = '  G. '


		for direction in policy:
			row = int(direction[0]/self.size[1])
			col = int(direction[0] - row*int(self.size[1]))
			direction_grid[row][col] = self.mapping[direction[1]]

		for value in Q:
			row = int(value[0]/self.size[1])
			col = int(value[0] - row*int(self.size[1]))
			Q_grid[row][col] = value[1]


		for i in range(2*len(direction_grid)+1):
			row = []
			for j in range(2*len(direction_grid[0])+1):
				if (i % 2) == 1 & (j % 2) == 1:
					row.append(direction_grid[(i-1)/2][(j-1)/2])
				elif (j % 2) == 0:
					row.append('|')
				else:
					row.append('_')
			print ' '.join(row)
		print

		for i in range(2*len(Q_grid)+1):
			row = []
			for j in range(2*len(Q_grid[0])+1):
				if (i % 2) == 1 & (j % 2) == 1:
					try:
						val = Q_grid[(i-1)/2][(j-1)/2]
						sign = '+'*(val > 0) + '-'*(val<=0)
						val = str(np.abs(round(val,2)))
						row.append(sign + val)
					except:
						val = Q_grid[(i-1)/2][(j-1)/2]
						row.append(val)
				elif (j % 2) == 0:
					row.append('|')
				else:
					row.append('_____')
			print ' '.join(row)
		print
