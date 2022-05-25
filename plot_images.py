import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval

plt.rcParams.update({
    "axes.labelsize": 18,
    "axes.linewidth": 1.5,
    "axes.titlesize": 20,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "legend.shadow": True,
    "legend.fancybox": True,
    "legend.fontsize": 13.0,
    "legend.framealpha": 1.0,
    "legend.title_fontsize": 20.0,
    "text.usetex": True,
    "xtick.bottom": "on",
    "xtick.direction": "in",
    "xtick.labelsize": 18,
    "xtick.major.pad": 5,
    "xtick.major.size": 16,
    "xtick.major.width": 1.5,
    "xtick.minor.pad": 5,
    "xtick.minor.size": 15,
    "xtick.minor.width": 1.5,
    "xtick.top": "on",
    "ytick.direction": "in",
    "ytick.left": "on",
    "ytick.labelsize": 18,
    "ytick.major.pad": 5,
    "ytick.major.size": 16,
    "ytick.major.width": 1.5,
    "ytick.minor.pad": 5,
    "ytick.minor.size": 15,
    "ytick.minor.width": 1.5,
    "ytick.right": "on",
})

colors_bi = np.array(['#F1C40F', '#34495E'])
cmap = plt.cm.get_cmap('Dark2');

nc = 8

# nc = 8
# colors_bi = np.array(['wheat', 'blue'])
# cmap = plt.cm.get_cmap('Accent')

print('Getting Time-based Features')
seconds = 10*60
folder = './results/time_final/'
features_path = f'{folder}/data/features_window_time_{seconds}.csv'
features = pd.read_csv(features_path)
data_cl_time = features[['ma_t_acceleration', 'msum_t_roc']]
file_name = f'{folder}/fishing_{nc}_{seconds}.csv'
dataset_time = pd.read_csv(file_name)


fig = plt.figure(figsize=(10, 9))
plt.scatter(data_cl_time['ma_t_acceleration'], data_cl_time['msum_t_roc'],
            c=colors_bi[dataset_time['labels_pos']], alpha=0.25)
plt.ylabel('\\textbf{Moving Sum --} Rate of Course${}_\mathrm{~over~Ground}$')
plt.xlabel('\\textbf{Moving Average --} Acceleration')
plt.yticks(np.arange(-500, 1100, step=100))
plt.xticks(np.arange(-1.4, 1.5, step=0.2))
plt.ylim(-500, 1000); plt.xlim(-1.2, 1.4)
plt.tight_layout()
plt.savefig(f'./results/images/time_scatter.pdf', bbox_inches='tight')
plt.savefig(f'./results/images/time_scatter.png', bbox_inches='tight', dpi=300)

fig = plt.figure(figsize=(10, 9))
plt.scatter(data_cl_time['ma_t_acceleration'], data_cl_time['msum_t_roc'],
            c=cmap(dataset_time['labels']), alpha=0.25)
plt.ylabel('\\textbf{Moving Sum (MS):} Rate of Course${}_\mathrm{~over~Ground}$')
plt.xlabel('\\textbf{Moving Average (MA):} Acceleration')
plt.yticks(np.arange(-600, 1100, step=100))
plt.xticks(np.arange(-1.3, 1.4, step=0.2))
plt.ylim(-500, 1000); plt.xlim(-1.1, 1.3)
plt.tight_layout()
plt.savefig(f'./results/images/time_scatter_{nc}.pdf', bbox_inches='tight')
plt.savefig(f'./results/images/time_scatter_{nc}.png', bbox_inches='tight', dpi=300)

print('Getting Obs-based Features')
win = 10
folder = './results/observations_final/'
features_path = f'{folder}/data/features_window_{win}.csv'
features = pd.read_csv(features_path)
data_cl_obs = features[['ma_acceleration', 'msum_roc']]
file_name = f'{folder}/fishing_{nc}_{win}.csv'
dataset_obs = pd.read_csv(file_name)

fig = plt.figure(figsize=(10, 9))
plt.scatter(data_cl_obs['ma_acceleration'], data_cl_obs['msum_roc'],
            c=colors_bi[dataset_obs['labels_pos']], alpha=0.25)
plt.ylabel('\\textbf{Moving Sum (MS):} Rate of Course${}_\mathrm{~over~Ground}$')
plt.xlabel('\\textbf{Moving Average (MA):} Acceleration')
plt.yticks(np.arange(-800, 1600, step=100))
plt.xticks(np.arange(-1.0, 0.8, step=0.2))
plt.ylim(-800, 1500); plt.xlim(-.6, 0.6)
plt.tight_layout()
plt.savefig(f'./results/images/obs_scatter.pdf', bbox_inches='tight')
plt.savefig(f'./results/images/obs_scatter.png', bbox_inches='tight', dpi=300)

fig = plt.figure(figsize=(10, 9))
plt.scatter(data_cl_obs['ma_acceleration'], data_cl_obs['msum_roc'],
            c=cmap(dataset_obs['labels']), alpha=0.25)
plt.ylabel('\\textbf{Moving Sum (MS):} Rate of Course${}_\mathrm{~over~Ground}$')
plt.xlabel('\\textbf{Moving Average (MA):} Acceleration')
plt.yticks(np.arange(-800, 1600, step=100))
plt.xticks(np.arange(-1.0, 0.8, step=0.2))
plt.ylim(-800, 1500); plt.xlim(-.6, .6)
plt.tight_layout()
plt.savefig(f'./results/images/obs_scatter_{nc}.pdf', bbox_inches='tight')
plt.savefig(f'./results/images/obs_scatter_{nc}.png', bbox_inches='tight', dpi=300)

print('Getting Dist-based Features')
km = 5.0
folder = './results/distance_final/'
features_path = f'{folder}/data/features_window_{km}.csv'
features = pd.read_csv(features_path)
data_cl_dist = features[['ma_d_acceleration', 'msum_d_roc']]
file_name = f'{folder}/fishing_{nc}_{km}.csv'
dataset_dist = pd.read_csv(file_name)

fig = plt.figure(figsize=(10, 9))
plt.scatter(data_cl_dist['ma_d_acceleration'], data_cl_dist['msum_d_roc'],
            c=colors_bi[dataset_dist['labels_pos']], alpha=0.25)
plt.ylabel('\\textbf{Moving Sum (MS):} Rate of Course${}_\mathrm{~over~Ground}$')
plt.xlabel('\\textbf{Moving Average (MA):} Acceleration')
plt.yticks(np.arange(-800, 1600, step=100))
plt.xticks(np.arange(-1.0, 0.8, step=0.2))
plt.ylim(-800, 1500); plt.xlim(-.6, 0.6)
plt.tight_layout()
plt.savefig(f'./results/images/dist_scatter.pdf', bbox_inches='tight')
plt.savefig(f'./results/images/dist_scatter.png', bbox_inches='tight', dpi=300)


fig = plt.figure(figsize=(10, 9))
plt.scatter(data_cl_dist['ma_d_acceleration'], data_cl_dist['msum_d_roc'],
            c=cmap(dataset_dist['labels']), alpha=0.25)
plt.ylabel('\\textbf{Moving Sum (MS):} Rate of Course${}_\mathrm{~over~Ground}$')
plt.xlabel('\\textbf{Moving Average (MA):} Acceleration')
plt.yticks(np.arange(-800, 1600, step=100))
plt.xticks(np.arange(-1.0, 0.8, step=0.2))
plt.ylim(-800, 1500); plt.xlim(-.6, .6)
plt.tight_layout()
plt.savefig(f'./results/images/dist_scatter_{nc}.pdf', bbox_inches='tight')
plt.savefig(f'./results/images/dist_scatter_{nc}.png', bbox_inches='tight', dpi=300)


###############
print('DBI plot')

eval_obs = pd.read_csv(f'./results/observations/measures.csv', index_col=0)
eval_time = pd.read_csv(f'./results/time/measures.csv', index_col=0)
eval_dist = pd.read_csv(f'./results/dist/measures2.csv', index_col=0)

seconds = 10 * 60
km = 5
obs_d = eval_obs.loc[:, str(10)].apply(lambda x: literal_eval(x)[0])
time_d = eval_time.loc[:, str(seconds)].apply(lambda x: literal_eval(x)[0])
dist_d = eval_dist.loc[:, str(km)].apply(lambda x: literal_eval(x)[0])
index_d = eval_obs.index

fig = plt.figure(figsize=(10, 7))
plt.plot(index_d, time_d, marker="s", linestyle="--", linewidth=2,
         markersize=7, label='Time')
plt.plot(index_d, dist_d, marker="o", linestyle="-.", linewidth=2,
         markersize=7, label='Distance')
plt.plot(index_d, obs_d, marker="d", linestyle="-", linewidth=2,
         markersize=7, label='Message')
plt.ylabel('\\texttt{$\lambda\'s$~DBI:} Davies–Bouldin Index')
plt.xlabel('$\lambda$: $k$-means using $\lambda$-clusters', fontsize=20)
plt.legend()
plt.xticks(index_d, index_d)
plt.yticks(np.arange(0.35, 0.55, step=0.01))
plt.tight_layout()
plt.savefig(f'./results/images/dbi-kmeans.pdf', bbox_inches='tight')
plt.savefig(f'./results/images/dbi-kmeans.png', bbox_inches='tight', dpi=300)

