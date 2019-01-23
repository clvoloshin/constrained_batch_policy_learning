import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns; sns.set(color_codes=True)
from matplotlib.ticker import FuncFormatter
def percent(x, pos):
    return '%1d%%' % (x)
percent_formatter = FuncFormatter(percent)

# Colors
alpha = 0.15
sns.set(style="whitegrid", palette="Paired")

colorSet = sns.color_palette("Paired", 10);
def color_gen():
    

    colors = [ "dusty purple", "faded green", "amber", "windows blue", "coral"]
    colors = sns.xkcd_palette(colors)
    idx  = -1
    while 1:
        idx = (idx + 1) % len(colors)
        yield colors[idx]


path = os.path.join(os.getcwd(), 'experimental_results')
files = os.listdir(path)
csvs = [f for f in files if 'fqe_quality' in f]

# tmp = pd.DataFrame([csv.split('.csv')[0].split('_')[2:] for csv in csvs], columns=['year','month','day','hour','minute','a','b'])
# results_file = 'fqe_quality_' + '_'.join(tmp.sort_values(by=['year','month','day','hour','minute'], ascending=False).iloc[0]) + '.csv'
# results_file = 'fqe_quality_2018_12_23_11_00_g_cnn.csv'
# dr_fix = 'fqe_quality_fixed_dr.csv'
results_file = 'fqe_quality_fixed_dr_tabular_4.csv'
df = pd.read_csv(os.path.join(path, results_file))
df['trial_num'] = np.array([[i]*10 for i in range(int(1+max(df['trial_num'])))]).reshape(-1)
df['num_trajectories'] = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9]*int(max(df['trial_num'])+1)
# df_dr_fix = pd.read_csv(os.path.join(path, dr_fix))

# df = df.merge(df_dr_fix, left_on=['epsilon','num_trajectories','trial_num'], right_on=['epsilon','num_trajectories','trial_num'], how='left')
# for col in [col for col in df.columns if '_y' in col]: 
#     if 'doubly_robust' not in col: 
#         del df[col]

# for col in [col for col in df.columns if ('_x' in col) and ('doubly_robust' in col)]:
#     del df[col]
# df.columns = [col.replace('_x', '') for col in df.columns]
# df.columns = [col.replace('_y', '') for col in df.columns]

def custom_plot(x, y, minimum, maximum, plot_band=True, zorder=11, alpha=.15, **kwargs):
    ax = kwargs.pop('ax', plt.gca())
    base, = ax.plot(x, y, **kwargs)
    if plot_band:
        ax.fill_between(x, minimum, maximum, facecolor=base.get_color(), alpha=alpha, zorder=zorder)

for epsilon, group in df.groupby('epsilon'):
    del group['epsilon']
    # group.set_index('num_trajectories').plot()
    # import pdb; pdb.set_trace()
    small_value = 1e-10
    exact = group['approx_pdis'].iloc[0]+small_value
    print list(group.apply(lambda x: x+exact).groupby('num_trajectories'))[-1][1][['trial_num', 'exact', 'fqe']]
    means = group.apply(lambda x: x+exact).groupby('num_trajectories').mean()
    stds = group.apply(lambda x: x+exact).groupby('num_trajectories').std()

    medians = group.apply(lambda x: x+exact).groupby('num_trajectories').median()
    lower_quants = group.apply(lambda x: x+exact).groupby('num_trajectories').quantile(.25)
    upper_quants = group.apply(lambda x: x+exact).groupby('num_trajectories').quantile(.75)

    del means['trial_num']
    del stds['trial_num']
    del medians['trial_num']
    del lower_quants['trial_num']
    del upper_quants['trial_num']

    print '*'*20
    print 'Epsilon: %s' % epsilon
    print means
    print stds

    fig, ax = plt.subplots(1)
    colors = color_gen()
    for i, col in enumerate(['fqe', 'approx_pdis', 'doubly_robust', 'weighted_doubly_robust']):
        # import pdb; pdb.set_trace()

        x = np.array(np.unique(group['num_trajectories']))
        mu = np.array(means[col])
        sigma = np.array(stds[col])
        lower_bound = mu + sigma
        upper_bound = mu - sigma
        # mu = np.array(medians[col])
        # lower_bound = np.array(lower_quants[col])
        # upper_bound = np.array(upper_quants[col])

        

        col = ['Fitted Q Evaluation (FQE)', 'Per-Decision IS (PDIS)', 'Doubly Robust (DR)', 'Weighted Doubly Robust (WDR)', 'AM'][i]
        if (i == 0) or (i == 3):
            custom_plot(x*100, mu, lower_bound, upper_bound, plot_band=True,zorder=11, marker='o', label=col,  color=colors.next())
        else:
            custom_plot(x*100, mu, lower_bound, upper_bound, plot_band=False,zorder=11, marker='o', label=col,  color=colors.next())
    
    custom_plot(x*100, [exact]*len(x), lower_bound, upper_bound, plot_band=False, marker='o', label='True Value',  color=colors.next())


    # means.plot(yerr=stds)

    # plt.title(epsilon)
    col = color_gen().next()
    print 'Number of Trials: ', max(df['trial_num'])+1
    ax.legend(loc='upper right')
    ax.grid(alpha=.35)
    # ax.set_title('Probability of exploration: %s' % epsilon)
    ax.set_xlabel('Percentage of Data Sub-Sampled for Evaluation')
    ax.set_ylabel('Estimated Constraint Value')
    ax.set_title('Off-Policy Evaluation - Standalone Comparison', fontsize=16)
    ax.xaxis.set_major_formatter(percent_formatter)
    ax.set_ylim(bottom=-1, top=0)
    plt.tight_layout()
    plt.savefig('lake_fqe_vs_others.png', format='png', dpi=300)
    plt.show()

