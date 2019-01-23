import deepdish as dd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.lines as mlines
from matplotlib.legend import Legend
import seaborn as sns; sns.set(color_codes=True)
import os
import scipy.signal as signal

# Colors
alpha = 0.15
sns.set(style="whitegrid", palette="Paired")
colorSet = sns.color_palette("Paired", 10);

def discounted_sum(costs, discount):
    '''
    Calculate discounted sum of costs
    '''
    y = signal.lfilter([1], [1, -discount], x=costs[::-1])
    return y[::-1][0]

def color_gen():
	
	colors = [ "dusty purple", "windows blue",  "faded green", "dark pink", "amber"]
	colors = sns.xkcd_palette(colors)
	idx  = -1
	while 1:
		idx = (idx + 1) % len(colors)
		yield colors[idx]

# Data setup 

dones = dd.io.load(os.path.join('seed_2_data', 'car_data_is_done_seed_2.h5'))
costs = dd.io.load(os.path.join('seed_2_data', 'car_data_rewards_seed_2.h5'))
dones = np.hstack([0,1+np.where(dones)[0]])
episodes = []
for low_, high_ in zip(dones[:-1], dones[1:]):
    new_episode ={
        'c': costs[low_:high_, 0].reshape(-1),
        'brake': costs[low_:high_, -1].reshape(-1),
        'lane': costs[low_:high_, 3].reshape(-1),
    }
    
    episodes.append(new_episode)

discounted_costs = np.array([[discounted_sum(x['c'],.95),discounted_sum(x['brake'],.95),discounted_sum(x['lane'],.95)]  for x in episodes])
data = dd.io.load('car_policy_improvement.h5')
DQN = [-39.61397106365249, 7.703194041056963, 115.62071639160499]
LSPI = pd.read_csv('lspi_results.csv')
plt.rc('text', usetex=True)


lines, fill_betweens= [], []
plt.rc_context({'axes.edgecolor':'k'})




fig = plt.figure(figsize=(12, 6))
grid = plt.GridSpec(6, 4, wspace=0.6, hspace=0.5)
ax1 = fig.add_subplot(grid[0:4, :2])



# fig, ax1 = plt.subplots()
ax1.grid(alpha=.35)
max_iterations = 27
iterations = range(len(data['g_eval'][0][:max_iterations]))
colors = color_gen()
constraint_names = ['Braking', 'Center of Lane']
constraint_upper_bound = [5.8, 85.]#[1.5, 5.]
locations = ['lower left', 'lower center', 'lower right']
fontsize = 16
legend_fontsize = 16
legend_title_fontsize = 16
major_tick_mark_size = 14


def derandomize(data, constraints, min_iteration):
	
	fqe_c = np.array(data['c_eval'][0])[:,-1]
	fqe_g_0 = np.array(data['g_eval'][0])[:,-1]
	fqe_g_1 = np.array(data['g_eval'][1])[:,-1]
	out = []
	for iteration in range(min_iteration, len(fqe_c)):

		df_tmp = pd.DataFrame(np.hstack([np.arange(min_iteration,iteration+1).reshape(1,-1).T, fqe_c[min_iteration:(iteration+1)].reshape(1,-1).T, fqe_g_0[min_iteration:(iteration+1)].reshape(1,-1).T, fqe_g_1[min_iteration:(iteration+1)].reshape(1,-1).T ]), columns=['iteration', 'fqe_c', 'fqe_g_0', 'fqe_g_1'])
		df_tmp = df_tmp[(df_tmp['fqe_g_0'] < constraints[0]) & (df_tmp['fqe_g_1'] < constraints[1]) ]
		try:
			argmin = np.argmin(np.array(df_tmp['fqe_c']))
			it = int(df_tmp.iloc[argmin]['iteration'])
		except:
			argmin = 0
			it = 0
		out.append(np.hstack([iteration, np.hstack([data['c_exacts'][it],  np.array(data['g_exacts'])[it,:-1]]) ]))

	return pd.DataFrame(out, columns=['iterations', 'c_derandomized', 'g_0_derandomized', 'g_1_derandomized'])


df_derandom = derandomize(data, np.array(constraint_upper_bound)*.8, 0)

legend = []
car_color = colors.next()
derandom_color = colors.next()
c_values = np.array(data['c_eval_actuals'])[:max_iterations,-10:,:] # shape = (iteration #, k, performance)
last = np.cumsum(c_values[:,-1,0])/np.arange(1,1+len(c_values[:,-1,0]))#*100
evaluation = np.array(pd.DataFrame(c_values[:,-1,0]).expanding().mean()).reshape(-1)
std = np.array(pd.DataFrame(c_values[:,-1,0]).expanding().std()).reshape(-1)

#evaluation = np.mean(c_values, axis=1)[:,0]#*100
#std = np.std(c_values, axis=1)[:,0]#*100
lines.append( ax1.plot(iterations, last, color = car_color,linestyle='-',markersize=7, label='Exact') )

lines.append( ax1.plot(df_derandom['iterations'], df_derandom['c_derandomized'], color = derandom_color,linestyle='-',markersize=7, label='Exact') )

# lines.append( ax1.plot(iterations, evaluation, color = car_color,marker='s',markersize=7, label='Mean of last 10') )
# fill_betweens.append( ax1.fill_between(iterations,evaluation-std, evaluation+std, alpha = alpha, color = car_color, zorder = 10) )

y_err_lower = last - np.min(c_values, axis=1)[:,0]
y_err_higher = last - np.max(c_values, axis=1)[:,0]
legend.append( mlines.Line2D([], [], color=car_color, linestyle='-',
                      markersize=7, label='Percent of Track Covered') )
legend.append( mlines.Line2D([], [], color=derandom_color, linestyle='-',
                      markersize=7, label='Percent of Track Covered') )
# legend.append( mlines.Line2D([], [], color=car_color, marker='s',
#                       markersize=7, label='Percent of Track Covered') )

## Baselines
	# LSPI
lspi_color = colors.next()
lspi = np.array(LSPI.iloc[:,0])
evaluation = np.array(pd.DataFrame(lspi).expanding().mean()).reshape(-1)
lines.append( ax1.plot(iterations, evaluation, color = lspi_color,markersize=7,linestyle='-' , label='Exact') )
legend.append( mlines.Line2D([], [], color=lspi_color, linestyle='-' ,
                      markersize=7, label='Percent of Track Covered') )

	# DQN
dqn_color = colors.next()
dqn = np.array([DQN[0]]*len(last))
evaluation = np.array(pd.DataFrame(dqn).expanding().mean()).reshape(-1)
lines.append( ax1.plot(iterations, evaluation, color = dqn_color,markersize=7,linestyle='-' , label='Exact') )
legend.append( mlines.Line2D([], [], color=dqn_color, linestyle='-' ,
                      markersize=7, label='Percent of Track Covered') )

	# Pi_D
pi_d_color = colors.next()
evaluation = np.mean(discounted_costs[:,0]).reshape(-1)
lines.append( ax1.plot(iterations, [evaluation]*len(iterations), color = pi_d_color,markersize=7,linestyle='-' , label='Exact') )
legend.append( mlines.Line2D([], [], color=pi_d_color, linestyle='-' ,
                      markersize=7, label='Percent of Track Covered') )

legend.append( mlines.Line2D([], [], color='k', linestyle='--' ,
                      markersize=7, label='Percent of Track Covered') )

ax1.set_xlabel('Iteration (t)', fontsize=fontsize)
ax1.set_ylabel('Value (Main Objective)', color='k', fontsize=fontsize+2)
ax1.tick_params(axis='y', labelcolor='k')
ax1.set_ylim(bottom=-55, top=-15)
# ax1.set_ylim(bottom=20, top=55)
ax1.set_xlim(-.5, 25)
labels = np.array(['FQE', 'Algo 2', 'Mean', 'Regularized LSPI', 'Online-RL \n(no constraint)', 'Algo 2 \n(Derandomized)', r'$\pi_D$', 'Constraint Threshold'])
leg = Legend(ax1, 
			np.array(legend)[[0,1,3,2,4,5]], 
			labels[[1,5,4,3,6,7]], 
			loc='lower left', 
			bbox_to_anchor=(.05,.02),
			bbox_transform=fig.transFigure,
			ncol = 2,
			frameon=True, 
			fontsize = legend_fontsize-1)
ax1.add_artist(leg)

plt.tick_params(axis='both', which='major', labelsize=major_tick_mark_size)
plt.tick_params(axis='both', which='minor', labelsize=8)
ax1.set_title('Main Objective - Accumulated Cost', fontsize=fontsize+2)
# plt.subplots_adjust(right=0.7)
# plt.tight_layout()#rect=[0,.2,1,1])
# plt.tight_layout(rect=[-.025,-.025,.675,1.025])
# plt.savefig('car_main_value_wo_band.png', format='png', dpi=300)
# plt.show()
# import pdb; pdb.set_trace()




# car_color = colors.next()
# c_values = np.array(data['c_eval'][0])[:,-10:] # shape = (iteration #, k)
# last = np.mean(c_values, axis=1)#c_values[:,-1,0]#*100
# evaluation = np.mean(c_values, axis=1)#*100
# std = np.std(c_values, axis=1)#*100
# lines.append( ax1.plot(iterations, last, color = car_color,marker='o',markersize=7, label='Percent of Track Covered') )
# fill_betweens.append( ax1.fill_between(iterations,evaluation-std, evaluation+std, alpha = alpha, color = car_color, zorder = 10) )
# legend.append( mlines.Line2D([], [], color=car_color, marker='o',
#                       markersize=7, label='Percent of Track Covered') )

#labels = np.array([r"$FQE: \;\; \frac{1}{10}\sum_{i=40}^{50}\widehat{C^{i}}(\pi_{Q_{50}})$", r"$Exact: \;\; C(\pi_{Q_{50}})$", r'$Mean: \;\; \frac{1}{10}\sum_{i=40}^{50} C(\pi_{Q_i})$'])
# labels = np.array(['FQE', r'$Our \; C$', 'Mean', r'$DQN \; C$', r'$LSPI \; C$'])
# leg = Legend(ax1, np.array(legend)[[0,1,2]], labels[[1,3,4]], title='Main Objective', loc=locations[0], frameon=False, fontsize = legend_fontsize) #shadow=True, fancybox=True, 
# ax1.add_artist(leg)
# ax = ax1.twinx()

# tex_labels = [[r'$FQE: \;\; \frac{1}{10}\sum_{i=40}^{50} \widehat{G^{i}_0}(\pi_{Q_{50}})$', r'$Exact: \;\; G_0(\pi_{Q_{50}})$'],
# 			  [r'$FQE: \;\; \frac{1}{10}\sum_{i=40}^{50} \widehat{G^{i}_1}(\pi_{Q_{50}})$', r'$Exact: \;\; G_1(\pi_{Q_{50}})$']]
# tex_labels = [[r'$FQE \; G_0$', r'$Our \; G_0$'], [r'$FQE \; G_1$', r'$Our \; G_1$']]
# baseline_labels = [[r'$DQN \; G_0$', r'$LSPI \; G_0$'], [r'$DQN \; G_1$', r'$LSPI \; G_1$']]
tex_labels = [[r'$FQE \; G_0$', 'Algo 2', 'Algo 2 (Derandomized)'], [r'$FQE \; G_1$', 'Algo 2', 'Algo 2 (Derandomized)']]
# baseline_labels = [['DDQN', 'LSPI'], ['DDQN', 'LSPI']]
baseline_labels = [['Online-RL (no constraint)', 'Regularized LSPI', r'$\pi_D$'], ['Online-RL (no constraint)', 'Regularized LSPI', r'$\pi_D$']]
# plt.clf()
# fig, axs = plt.subplots(2, sharex=True)
axs = []
axs.append(fig.add_subplot(grid[:3, 2:]))
axs.append(fig.add_subplot(grid[3:, 2:]))


for idx in data['g_eval'].keys():
	colors = color_gen()
	ax = axs[idx]
	ax.grid(alpha=.35)
	legend = []
	# FQE
	constraint = np.array(data['g_eval'][idx]) # shape = (iteration #, k) referring to Q_k
	constraint = constraint[:max_iterations,-10:]
	
	# evaluation = np.array(pd.DataFrame(np.mean(constraint, axis = 1)/constraint_upper_bound[idx]).expanding().mean()).reshape(-1)
	# std = np.array(pd.DataFrame(np.mean(constraint, axis = 1)/constraint_upper_bound[idx]).expanding().std()).reshape(-1)

	# evaluation = np.mean(constraint, axis = 1)/constraint_upper_bound[idx]
	# std = np.std(constraint, axis=1)/constraint_upper_bound[idx]
	
	label = constraint_names[idx]
	
	# lines.append( ax.plot(iterations, evaluation, color = color,marker='o',markersize=7, label=tex_labels[idx][0]) )
	# fill_betweens.append( ax.fill_between(iterations,evaluation-std, evaluation+std, alpha = alpha, color = color, zorder = 11+2*idx) )
	
	#legend.append( mlines.Line2D([], [], color=color, marker='o',
    #                      markersize=7, label=tex_labels[idx][0]) )

    # EXACT
	g_exacts = np.array(data['g_exacts'])[:max_iterations,idx]#/constraint_upper_bound[idx]
	evaluation = np.array(pd.DataFrame(g_exacts).expanding().mean()).reshape(-1)
	std = np.array(pd.DataFrame(g_exacts).expanding().std()).reshape(-1)

	lines.append( ax.plot(iterations, evaluation, color = car_color ,linestyle='-' ,linewidth=2.0, label=tex_labels[idx][1])  )
	legend.append( mlines.Line2D([], [], color=car_color, linestyle='-' ,linewidth=2.0,
                          markersize=7, label=tex_labels[idx][1]) )

	# Derandomized
	lines.append( ax.plot(df_derandom['iterations'], df_derandom['g_%s_derandomized' % idx], linewidth=2.0,color = derandom_color,linestyle='-',markersize=7, label='Exact') )
	legend.append( mlines.Line2D([], [], color=derandom_color, linestyle='-' ,linewidth=2.0,
                          markersize=7, label=tex_labels[idx][2]) )


	# fill_betweens.append( ax.fill_between(iterations,evaluation-std, evaluation+std, alpha = alpha, color = color, zorder = 11+2*idx+1) )

	## BASELINES

	# fill_betweens.append( ax.fill_between(iterations,evaluation-std, evaluation+std, alpha = alpha, color = color, zorder = 11+2*idx+1) )
		# LSPI
	baseline = np.array(LSPI.iloc[:,idx+1])*constraint_upper_bound[idx]
	evaluation = np.array(pd.DataFrame(baseline).expanding().mean()).reshape(-1)
	std = np.array(pd.DataFrame(baseline).expanding().std()).reshape(-1)

	lines.append( ax.plot(iterations, evaluation, color = lspi_color , linestyle='-', linewidth=2.0,label=baseline_labels[idx][1])  )
	legend.append( mlines.Line2D([], [], color=lspi_color, linestyle='-', linewidth=2.0,
                          markersize=7, label=baseline_labels[idx][1]) )
	# fill_betweens.append( ax.fill_between(iterations,evaluation-std, evaluation+std, alpha = alpha, color = color, zorder = 11+2*idx+1) )
		# DQN
	baseline = np.array([DQN[idx+1]]*len(evaluation))#/constraint_upper_bound[idx]
	evaluation = np.array(pd.DataFrame(baseline).expanding().mean()).reshape(-1)
	std = np.array(pd.DataFrame(baseline).expanding().std()).reshape(-1)

	lines.append( ax.plot(iterations, evaluation, color = dqn_color , linestyle='-' ,linewidth=2.0, label=baseline_labels[idx][0])  )
	legend.append( mlines.Line2D([], [], color=dqn_color, linestyle='-',linewidth=2.0,
                          markersize=7, label=baseline_labels[idx][0]) )

		# Pi_D
	evaluation = np.mean(discounted_costs[:,idx+1]).reshape(-1)
	lines.append( ax.plot(iterations, [evaluation]*len(iterations), color = pi_d_color,markersize=7,linewidth=2.0,linestyle='-' , label=baseline_labels[idx][2]) )
	legend.append( mlines.Line2D([], [], color=pi_d_color, linestyle='-' ,linewidth=2.0,
                      markersize=7, label=baseline_labels[idx][2]) )


	# THRESHOLD
	constraint_violation = [constraint_upper_bound[idx]]*len(iterations)
	lines.append( ax.plot(iterations, constraint_violation, color = 'k', linestyle=':', linewidth=2.0, marker=None) )
	legend.append( mlines.Line2D([], [], color='k', linestyle=':', marker=None,  linewidth=2.0, label='Constraint Threshold') )


	labels = []
	for i in range(len(legend)):
		line = legend[i]
		try:
			labels += [line.label]
		except:
			labels += [line.get_label()]
	
	if idx == 1:
		# leg = Legend(ax, 
		# 			legend, 
		# 			labels, 
		# 			title=label, 
		# 			# loc='center left', 
		# 			bbox_to_anchor=(0.5, -0.05),
		# 			ncol = 3,
		# 			frameon=False, 
		# 			# bbox_to_anchor=(1, 0.5), 
		# 			fontsize = legend_fontsize-3)
		leg = Legend(ax, 
			legend, 
			labels,
			loc='lower center', 
			bbox_to_anchor=(.5,0),
			bbox_transform=fig.transFigure,
			ncol = 2,
			frameon=True, 
			fontsize = legend_fontsize-2)

		plt.setp(leg.get_title(),fontsize='%s' % legend_title_fontsize)
		# ax.add_artist(leg)

	if idx == 1:
		ax.set_xlabel('Iteration (t)', fontsize=fontsize)
	lab = ['Value (Braking)', 'Value (Lane Center)'][idx]
	ax.set_ylabel(lab, color='k', fontsize=fontsize+2)
	# ax.set_ylim(bottom=-1, top=3)
	ax.tick_params(axis='y', labelcolor='k')
	if idx == 0: ax.set_ylim(bottom=-2, top=15)
	ax.set_xlim(-.5, 25)
axs[0].tick_params(axis='both', which='major', labelsize=major_tick_mark_size)
axs[0].set_xticklabels([])
axs[1].tick_params(axis='both', which='major', labelsize=major_tick_mark_size)
axs[0].set_title('Constraints - Accumulated Cost', fontsize=fontsize+2)
plt.tight_layout(rect=[-.025,-.025,1.025,1.025])#rect=[0,.2,1,1])
plt.savefig('car_all_values_wo_band.png', format='png', dpi=300)
plt.show()

import pdb; pdb.set_trace()


