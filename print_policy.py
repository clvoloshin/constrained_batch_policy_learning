import numpy as np

class PrintPolicy(object):
	def __init__(self, size=[4,4]):
		self.mapping = {0:'<', 1:'v', 2:'>', 3:'^'}
		self.size = size
		self.action_space_dim = len(self.mapping.keys())

	def pprint(self, *args):
		if len(args) == 1:
			pi = args[0]
			size = self.size[0]*self.size[1]
			if not isinstance(pi,(list,)):
				pi = [pi]

			if len(pi) == 0: return
	
		
			states = np.array(range(size)).reshape(1,-1).T
			actions_for_each_pi = np.hstack([[np.eye(self.action_space_dim)[p.min_over_a(np.arange(size))[1]] for p in pi]])
			policy = np.hstack([states, np.argmax(actions_for_each_pi.mean(0), 1).reshape(1,-1).T])

			Qs_for_each_pi = np.vstack([np.array([p.all_actions(np.arange(size))]) for p in pi])
			Q = np.hstack([states, np.mean(Qs_for_each_pi,axis=0)[np.arange(Qs_for_each_pi.shape[1]),policy[:,1]].reshape(-1,1)])
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
