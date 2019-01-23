
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.lines as mlines
from matplotlib.legend import Legend
import seaborn as sns
import deepdish as dd
from mpl_toolkits.axes_grid1 import make_axes_locatable
sns.set(context="paper")#, font="monospace")
plt.rc('text', usetex=True)
#sns.set(style="darkgrid", palette="Paired")

# Load the datset of correlations between cortical brain networks
#df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)
#corrmat = df.corr()

# matrix = np.load('policy_role_freq.npy')
import os
df = pd.read_csv(os.path.join(os.getcwd(),'experimental_results','lspi.csv'))
df = pd.DataFrame(np.array([[y.strip('(').strip(')') for y in x] for x in [x.split(',') for x in np.array(df.columns)]]).astype(float),
		columns = ['lambda_0', 'lambda_1', 'c_pi_exact', 'g_pi_exact_0', 'g_pi_exact_1', 'performance'])
# import pdb; pdb.set_trace()
df = pd.read_csv(os.path.join(os.getcwd(),'experimental_results','results_grid.csv'))
data = dd.io.load(os.path.join(os.getcwd(),'experimental_results','policy_improvement_grid.h5'))
# performance = np.array(df['performance']) 
performance = np.array(data['c_performance'])
df = df[['c_pi_exact','g_pi_exact_0','g_pi_exact_1','lambda_0','lambda_1']]
		#labels=['c_pi_exact','g_pi_exact_0','g_pi_exact_1','lambda_0','lambda_1','performance'])


main = np.array(df['c_pi_exact']).reshape(11,11)
braking = np.array(df['g_pi_exact_0']).reshape(11,11)
lane = np.array(df['g_pi_exact_1']).reshape(11,11)


# import pdb; pdb.set_trace()


# # Set up the matplotlib figure
# f, axarr = plt.subplots(nrows=1,ncols=3, figsize=(12, 9))
# sns.set(font_scale=2)
# upper_bound = [-60, 8, 135.]#[1.5, 5.]
# lower_bound = [-30, 0., 0.]#[1.5, 5.]
# for i, matrix in enumerate([main, braking, lane]):
# 	sns.heatmap(matrix, cmap = 'summer', ax=axarr[i], vmin= lower_bound[i], vmax =upper_bound[i], square=True)
# 	axarr[i].tick_params(axis='x', labelsize=18)
# 	axarr[i].tick_params(axis='y', labelsize=18)
# 	axarr[i].set_xlabel(r'$\lambda_0$' + ' (Braking Penalty)', fontsize = 18)
# 	axarr[i].set_ylabel(r'$\lambda_1$' + ' (Center of Lane Penalty)', fontsize = 18)



# #g.axes[0,0].set_xlabel('axes label 1')

# # Use matplotlib directly to emphasize known networks
# """
# networks = corrmat.columns.get_level_values("network")
# for i, network in enumerate(networks):
#     if i and network != networks[i - 1]:
#         ax.axhline(len(networks) - i, c="w")
#         ax.axvline(i, c="w")
# """
# f.tight_layout()
# #f.savefig('role_frequency.png', format='png', dpi=300)
# plt.show()








# generate data
constraints = [5.8, 85.]
use_rewards = True
# x = np.linspace(0,1, num=11)
# y = np.linspace(0,1, num=11)
# X,Y = np.meshgrid(x,y)
# signal = main.reshape(-1)
det = (braking.reshape(-1) < constraints[0]) & (lane.reshape(-1) < constraints[1]) & (performance.reshape(-1) >= .95)  #np.random.poisson(lam=0.5,size=len(x)*len(y))
det = det.astype(int)

df_signal = df[['c_pi_exact', 'lambda_0', 'lambda_1']]#pd.DataFrame({"y":df.flatten(), "x":X.flatten(), "intensity":signal})
df_signal.columns = ['intensity', 'x', 'y']
df_det = pd.DataFrame({"y":df['lambda_1'], "x":df['lambda_0'], "det":det})
df_signal['intensity'] = -use_rewards*df_signal['intensity']

# prepare Dataframes
dfmark = df_det[df_det["det"]>0]

#plotting
fig, ax=plt.subplots()
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)

x = df_signal["x"].unique()
y = df_signal["y"].unique()
ext = [x.min()-np.diff(x)[0]/2.,x.max()+np.diff(x)[0]/2., 
       y.min()-np.diff(y)[0]/2.,y.max()+np.diff(y)[0]/2. ]

# df_signal['y'] += 1
# df_signal['y'] = 1/df_signal['y']
df = df_signal.pivot(index="y", columns="x")
im = ax.imshow(df, extent=ext, cmap=plt.get_cmap('YlGnBu'), origin='lower')
ax.set_xticks(x)
ax.set_xticklabels([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], fontsize=14)
ax.set_yticks(y)
ax.set_yticklabels([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], fontsize=14)

# ax.scatter(dfmark["x"], dfmark["y"], marker="s", s=100, c="crimson")
dx = np.diff(x)[0]; dy = np.diff(y)[0]
recs = []
for (xi,yi), in zip(dfmark[["x","y"]].values):
    rec = plt.Rectangle((xi-dx/2.,yi-dy/2.),dx,dy, fill=False,
                        edgecolor="black", lw=2, hatch='\\')
    recs.append(rec)
    ax.add_artist(rec)

rec = plt.Rectangle((0.,0.),0,0, fill=False,
                        edgecolor="black", lw=2, hatch='\\')
recs.append(rec)

df = df_signal.merge(df_det, how='left')

# good_policies = -use_rewards*np.array(main).reshape(11,11) * det.reshape(11,11)
for (xi, yi) in df[df['det']>0][df[df['det'] > 0]['intensity'] == df[df['det'] > 0]['intensity'].max()][['x','y']].values:
    
    best = plt.Rectangle((xi-dx/2.,yi-dy/2.),dx,dy, fill=False,
                        edgecolor="crimson", lw=2, hatch='*' )
    ax.add_artist(best)

best = plt.Rectangle((0.,0.),0,0, fill=False,
                        edgecolor="crimson", lw=2, hatch='*' )

# plt.legend([recs[-1], best])
cbar = fig.colorbar(im, cax=cax, orientation='vertical', ticks=np.arange(-10, 60, 10)[::-1])
cbar.ax.set_yticklabels(np.arange(10, -60, -10)[::-1], fontsize=16)
cax.set_ylabel('Main Objective Value', fontsize=18)
ax.set_xlabel(r'$\lambda_0$' + ' (Braking Penalty)', fontsize = 18)
ax.set_ylabel(r'$\lambda_1$' + ' (Center of Lane Penalty)', fontsize = 18)
ax.legend([recs[-1], best], ['Satisfies Constraints', 'Best, Satisfies Constraints'], fontsize=16, loc='upper left', framealpha=.4)
ax.set_title('Regularized FQI Grid Search', fontsize=18)
plt.tight_layout()
plt.savefig('fqi_grid_search.png', format='png', dpi=300)
plt.show()


