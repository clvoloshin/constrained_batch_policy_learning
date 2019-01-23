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
from exponentiated_gradient import ExponentiatedGradient
from matplotlib.lines import Line2D

# Colors
alpha = 0.15
sns.set(style="whitegrid", palette="Paired")
colorSet = sns.color_palette("Paired", 10);

def color_gen():

	colors = ["dark pink","dusty purple", "amber", "faded green", "windows blue", ]
	colors = sns.xkcd_palette(colors)
	idx  = -1
	while 1:
		idx = (idx + 1) % len(colors)
		yield colors[idx]

plt.rc('text', usetex=True)
EG = ExponentiatedGradient(5., 2, 10.)

path = os.path.join(os.getcwd(), 'experimental_results')
files = os.listdir(path)
csvs = [f for f in files if 'experiment_results' in f]
tmp = pd.DataFrame([csv.split('.csv')[0].split('_')[2:] for csv in csvs], columns=['year','month','day','hour','minute'])
results_file = 'experiment_results_' + '_'.join(tmp.sort_values(by=['year','month','day','hour','minute'], ascending=False).iloc[0]) + '.csv'

# results_file = 'experiment_results_12_18_2018_22_20.csv'
df = pd.read_csv(os.path.join(path, results_file))
df['iteration'] -= 2

df['primal_dual_gap'] = df['max_L'] - df['min_L']

# plt.plot(df['iteration'], df['primal_dual_gap'])
# plt.show()
def unrandomize(df, constraints, min_iteration):
	df = df[df['iteration'] >= min_iteration]

	out = []
	for iteration in range(min_iteration, int(max(df['iteration']))):
		df_tmp = df[df['iteration'] <= iteration]
		df_tmp = df_tmp[df_tmp['g_pi'] < constraints[0]]
		argmin = np.argmin(np.array(df_tmp['c_pi']))
		out.append(np.hstack([iteration, np.array(df_tmp.iloc[argmin][['c_pi_exact', 'g_pi_exact']]) ]))

	return pd.DataFrame(out, columns=['iteration', 'c_unrandomized', 'g_unrandomized'])


df_unrandom = unrandomize(df, [.1], 5)


# f, ax = plt.subplots()
# ax.plot(df['iteration'], df['primal_dual_gap'])
# ax.set_title('Primal Dual Gap')

# import pdb; pdb.set_trace()

fontsize=20
colors = color_gen()
color_optimal = colors.next()
plt.plot(df['iteration'], df['primal_dual_gap'], color='b', label='Empirical Gap')
plt.plot(df['iteration'], [0]*len(df['iteration']), color=color_optimal, label='Minimum/Optimal Gap')
plt.xlabel('Iteration ' + r'$(t)$', fontsize=fontsize)
plt.ylabel('Primal-Dual Gap ' + r'$(\widehat{L}_{max} - \widehat{L}_{min})$', fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.tick_params(axis='both', which='major', labelsize=fontsize)
plt.xlim((-1,150))
plt.ylim((-.02,2))
plt.tight_layout()
plt.title('Convergence Behavior of Algo 2', fontsize=fontsize)
plt.savefig('lake_primal_dual_gap.png', format='png', dpi=300)
plt.clf()
plt.show()

# W BANDS
fontsize=16
f, axarr = plt.subplots(2, sharex=True)
colors = color_gen()
color_optimal = colors.next()
color_main = colors.next()
vals = pd.DataFrame(df['c_pi'])
evaluation = np.array(vals.expanding().mean()).reshape(-1)
std = np.array(vals.expanding().std()).reshape(-1)
axarr[0].plot(df['iteration'], evaluation, color=color_main, label='Algo 2')
axarr[0].fill_between(df['iteration'],evaluation-std, evaluation+std, alpha = alpha, color = color_main, zorder = 11)
color_pi_d = colors.next()
axarr[0].plot(df['iteration'], [-9.94428910084026e-05]*len(df['iteration']), color=color_pi_d, label=r'$\pi_D$') 
axarr[0].fill_between(df['iteration'],[-9.94428910084026e-05-0.002297397386833141]*len(df['iteration']), [-9.94428910084026e-05+0.002297397386833141]*len(df['iteration']), alpha = alpha, color = color_main, zorder = 11)
# axarr[0].set_ylabel('Main Objective Value \n of ' + r'$\widehat{\pi_t}$', fontsize=fontsize)
axarr[0].plot(df['iteration'], [-(.9**13)]*len(df['iteration']), color=color_optimal, label='Optimal Value')
axarr[0].set_ylabel('Main Objective Value', fontsize=fontsize-2)
line0 = Line2D([0,1],[0,1],linestyle='-', color=color_main)
line2 = Line2D([0,1],[0,1],linestyle='-', color=color_pi_d)
line4 = Line2D([0,1],[0,1],linestyle='-', color=color_optimal)
# axarr[0].legend([line0, line2, line4], ['Algo 2', r'$\pi_D$', 'Optimal Value'], loc='lower right', fontsize=12, frameon=True)
# axarr[0].legend(fontsize=fontsize, loc='lower right', frameon=True)
axarr[0].grid(alpha=.35)


evaluation = np.array(pd.DataFrame(df['g_pi_exact']).expanding().mean()).reshape(-1)
std = np.array(pd.DataFrame(df['g_pi_exact']).expanding().std()).reshape(-1)
axarr[1].plot(df['iteration'], evaluation, linewidth=2.0, linestyle=(0,[8,8]), color=color_main, label=r'$G(\widehat{\pi_t})$')
axarr[1].fill_between(df['iteration'],evaluation-std, evaluation+std, alpha = alpha, color = color_main, zorder = 11)
axarr[1].plot(df['iteration'], [0.15173932921437544]*len(df['iteration']), color=color_pi_d, label=r'$pi_D$') 
axarr[1].fill_between(df['iteration'],[0.15173932921437544-0.162341876715503]*len(df['iteration']),[0.15173932921437544+0.162341876715503]*len(df['iteration']), alpha = alpha, color = color_pi_d, zorder = 11)
axarr[1].plot(df['iteration'], [0]*len(df['iteration']), color=color_optimal, linewidth=2., linestyle=(8,[8,8]), label='Optimal value')
axarr[1].plot(df['iteration'], [.1]*len(df['iteration']), color = 'k', linestyle=':', linewidth=2.0, label='Threshold', marker=None)

line0 = Line2D([0,1],[0,1],linestyle='-', color=color_main)
line2 = Line2D([0,1],[0,1],linestyle='-', color=color_pi_d)
line4 = Line2D([0,1],[0,1],linestyle='-', color=color_optimal)
line5 = Line2D([0,1],[0,1],linestyle='--', color='k')
axarr[1].legend([line0, line2, line4, line5], ['Algo 2', r'$\pi_D$', 'Optimal Value', 'Constraint Threshold'], loc='lower center', bbox_to_anchor=(.5,0), bbox_transform=f.transFigure, ncol = 2, fontsize=fontsize, frameon=True)

# axarr[1].set_ylabel('Constraint Value \n of ' + r'$\widehat{\pi_t}$', fontsize=fontsize)
axarr[1].set_ylabel('Constraint Value', fontsize=fontsize-2)
# axarr[1].legend([line0, line1, line2, line3], ['Our Algorithm', 'DDQN (no constraint)', 'Optimal Value', 'Threshold'], loc='lower right', fontsize=fontsize, frameon=True)
axarr[1].grid(alpha=.35)
axarr[1].set_ylim(-.05, .35)

plt.xlabel('Iteration ' + r'$(t)$', fontsize=fontsize)
plt.xlim((-1,150))
# plt.ylim((-.02,2))
plt.tight_layout(rect=[0,.15,1,1])
fig = plt.gcf()
size = fig.get_size_inches()
# fig.set_size_inches(size[0], size[1]+.75)
axarr[0].set_title('Main Objective and Constraint -  Accumulated Cost', fontsize=fontsize)
plt.savefig('lake_values.png', format='png', dpi=300)
plt.show()

# WO BANDS
f, axarr = plt.subplots(2, sharex=True)
colors = color_gen()
color_optimal = colors.next()
color_main = colors.next()
vals = pd.DataFrame(df['c_pi'])
evaluation = np.array(vals.expanding().mean()).reshape(-1)
std = np.array(vals.expanding().std()).reshape(-1)
axarr[0].plot(df['iteration'], evaluation, color=color_main, label='Algo 2')
# axarr[0].fill_between(df['iteration'],evaluation-std, evaluation+std, alpha = alpha, color = color_main, zorder = 11)
color_pi_d = colors.next()
axarr[0].plot(df['iteration'], [-9.94428910084026e-05]*len(df['iteration']), color=color_pi_d, label=r'$\pi_D$') 
spacing = 8
axarr[0].plot(df['iteration'], [-(.9**13)]*len(df['iteration']), linestyle=(0*spacing,[spacing,spacing*2]), color=color_optimal, label='Optimal Value')
color_ddqn = colors.next()
axarr[0].plot(df['iteration'], [-(.9**13)]*len(df['iteration']), linestyle=(1*spacing,[spacing,spacing*2]), color=color_ddqn, label='DDQN (no constraint)')
color_unrandomized = colors.next()
# axarr[0].plot(df_unrandom['iteration'], df_unrandom['c_unrandomized'], linestyle=(16,[8,24]), color=color_unrandomized, label='Algo 2 (Unrandomized)')
axarr[0].plot(df['iteration'], [-(.9**13)]*len(df['iteration']), linestyle=(2*spacing,[spacing,spacing*2]), color=color_unrandomized, label='Algo 2 (Unrandomized)')

# axarr[0].set_ylabel('Main Objective Value \n of ' + r'$\widehat{\pi_t}$', fontsize=fontsize)
axarr[0].set_ylabel('Main Objective Value', fontsize=fontsize-2)
line0 = Line2D([0,1],[0,1],linestyle='-', color=color_main)
line1 = Line2D([0,1],[0,1],linestyle='-', color=color_unrandomized)
line2 = Line2D([0,1],[0,1],linestyle='-', color=color_pi_d)
line3 = Line2D([0,1],[0,1],linestyle='-', color=color_ddqn)
line4 = Line2D([0,1],[0,1],linestyle='-', color=color_optimal)
# axarr[0].legend([line0, line1, line2, line3, line4], ['Algo 2', 'Algo 2 (Derandomized)', r'$\pi_D$', 'Online-RL (no constraint)', 'Optimal Value' ], loc='lower right', fontsize=12, frameon=True)
# axarr[0].legend(fontsize=fontsize, loc='lower right', frameon=True)
axarr[0].grid(alpha=.35)


evaluation = np.array(pd.DataFrame(df['g_pi_exact']).expanding().mean()).reshape(-1)
std = np.array(pd.DataFrame(df['g_pi_exact']).expanding().std()).reshape(-1)
axarr[1].plot(df['iteration'], evaluation, linewidth=2.0, linestyle=(0*spacing,[spacing,spacing*3]), color=color_main, label=r'$G(\widehat{\pi_t})$')
# axarr[1].fill_between(df['iteration'],evaluation-std, evaluation+std, alpha = alpha, color = color_main, zorder = 11)
axarr[1].plot(df['iteration'], [0.15173932921437544]*len(df['iteration']), color=color_pi_d, label=r'$pi_D$') 
axarr[1].plot(df['iteration'], [0]*len(df['iteration']), color=color_optimal, linewidth=2.0, linestyle=(1*spacing,[spacing,spacing*3]), label='Optimal value')
axarr[1].plot(df['iteration'], [0]*len(df['iteration']), color=color_unrandomized, linewidth=2.0, linestyle=(2*spacing,[spacing,spacing*3]), label='Unrandomized')
axarr[1].plot(df['iteration'], [0]*len(df['iteration']), color=color_ddqn, linewidth=2.0, linestyle=(3*spacing,[spacing,spacing*3]), label='DDQN')
axarr[1].plot(df['iteration'], [.1]*len(df['iteration']), color = 'k', linestyle=':', linewidth=2.0, label='Threshold', marker=None)

line0 = Line2D([0,1],[0,1],linestyle='-', color=color_main)
line1 = Line2D([0,1],[0,1],linestyle='-', color=color_unrandomized)
line2 = Line2D([0,1],[0,1],linestyle='-', color=color_pi_d)
line3 = Line2D([0,1],[0,1],linestyle='-', color=color_ddqn)
line4 = Line2D([0,1],[0,1],linestyle='-', color=color_optimal)
line5 = Line2D([0,1],[0,1],linestyle='--', color='k')
axarr[1].legend([line0, line1, line2, line3, line4, line5], ['Algo 2', 'Algo 2 (Derandomized)', r'$\pi_D$', 'Online-RL (no constraint)', 'Optimal Value', 'Constraint Threshold'], loc='lower center', bbox_to_anchor=(.5,0), bbox_transform=f.transFigure, ncol = 2,  fontsize=fontsize-2, frameon=True)

# axarr[1].set_ylabel('Constraint Value \n of ' + r'$\widehat{\pi_t}$', fontsize=fontsize)
axarr[1].set_ylabel('Constraint Value', fontsize=fontsize-2)
# axarr[1].legend([line0, line1, line2, line3], ['Our Algorithm', 'DDQN (no constraint)', 'Optimal Value', 'Threshold'], loc='lower right', fontsize=fontsize, frameon=True)
axarr[1].grid(alpha=.35)

plt.xlabel('Iteration ' + r'$(t)$', fontsize=fontsize)
plt.xlim((-1,150))
# plt.ylim((-.02,2))
plt.tight_layout(rect=[0,.18,1,1])
fig = plt.gcf()
size = fig.get_size_inches()
axarr[0].set_title('Main Objective and Constraint -  Accumulated Cost', fontsize=fontsize)
# fig.set_size_inches(size[0], size[1]+.75)
plt.savefig('lake_values_wo_band.png', format='png', dpi=300)
plt.show()






# # Two subplots, the axes array is 1-d
# number_of_constraints = len([col for col in df.columns if 'g_avg_' in col])
# f, axarr = plt.subplots(2+number_of_constraints, sharex=True)
# axarr[0].plot(df['iteration'], df['primal_dual_gap'], label='gap')
# # axarr[0].plot(df['iteration'], pd.ewma(df['primal_dual_gap'], span=100), label='moving average')
# axarr[0].plot(df['iteration'], [0]*len(df['iteration']), color='g', label='Minimum/Optimal Gap')
# axarr[0].set_title('Primal Dual Gap')
# axarr[0].legend()
# axarr[1].plot(df['iteration'], df['c_avg'], color='b', label='C fqe')
# axarr[1].plot(df['iteration'], df['c_exact_avg'], color = 'r', label='C exact')
# # axarr[1].plot(df['iteration'], [-0.254186583]*len(df['iteration']), color='g', label='C optimal')
# # axarr[1].scatter(0, -2.763302804497763e-05, marker='x', color='k', label='C pi_old')
# axarr[1].set_title('Value of C of mean policy')
# axarr[1].legend()
# for col in range(number_of_constraints):
# 	axarr[2+col].plot(df['iteration'], df['g_avg_%s' % col], color='b', label='G_%s fqe' % col)
# 	axarr[2+col].plot(df['iteration'], df['g_exact_avg_%s' % col], color='r', label='G_%s exact' % col)
# 	# axarr[2].plot(df['iteration'], [0.]*len(df['iteration']), color='g', label='G optimal')
# 	# axarr[2].scatter(0, 0.13755537388963082, marker='x', color='k', label='G pi_old')
# 	axarr[2+col].set_title('Value of G_%s of mean policy' % col)
# 	axarr[2+col].legend()
# plt.show()

# # Number episodes achieved goal: 5. Number episodes fell in hole: 4890
# # C(pi_old): -7.560596707938992e-06. G(pi_old): 0.13777596062648703
# # Percentage of State/Action space seen: 0.943396226415

