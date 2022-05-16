import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval

nc = 8
colors_bi = np.array(['wheat', 'blue'])
cmap = plt.cm.get_cmap('Accent')

print('Getting Time-based Features')
seconds = 10*60
folder = './results/time_final/'
features_path = f'{folder}/data/features_window_time_{seconds}.csv'
features = pd.read_csv(features_path)
data_cl_time = features[['ma_t_acceleration', 'msum_t_roc']]
file_name = f'{folder}/fishing_{nc}_{seconds}.csv'
dataset_time = pd.read_csv(file_name)


fig = plt.figure(figsize=(10, 9))
plt.scatter(data_cl_time['ma_t_acceleration'], data_cl_time['msum_t_roc'], c=colors_bi[dataset_time['labels_pos']], alpha=0.7)
plt.xlabel('MA of acceleration', fontsize=15)
plt.ylabel('MS of ROC', fontsize=15)
plt.xticks(np.arange(-1.5, 1.7, step=0.2), fontsize=15)
plt.yticks(np.arange(-800, 1600, step=100), fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()
# plt.savefig(f'./results/images/time_scatter.png', bbox_inches='tight')

fig = plt.figure(figsize=(10, 9))
plt.scatter(data_cl_time['ma_t_acceleration'], data_cl_time['msum_t_roc'], c=cmap(dataset_time['labels']), alpha=0.7)
plt.xlabel('MA of acceleration', fontsize=15)
plt.ylabel('MS of ROC', fontsize=15)
plt.xticks(np.arange(-1.5, 1.7, step=0.2), fontsize=15)
plt.yticks(np.arange(-800, 1600, step=100), fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()
# plt.savefig(f'./results/images/time_scatter_{nc}.png', bbox_inches='tight')



print('Getting Obs-based Features')
win = 10
folder = './results/observations_final/'
features_path = f'{folder}/data/features_window_{win}.csv'
features = pd.read_csv(features_path)
data_cl_obs = features[['ma_acceleration', 'msum_roc']]
file_name = f'{folder}/fishing_{nc}_{win}.csv'
dataset_obs = pd.read_csv(file_name)

fig = plt.figure(figsize=(10, 9))
plt.scatter(data_cl_obs['ma_acceleration'], data_cl_obs['msum_roc'], c=colors_bi[dataset_obs['labels_pos']], alpha=0.7)
plt.xlabel('MA of acceleration', fontsize=15)
plt.ylabel('MS of ROC', fontsize=15)
plt.xticks(np.arange(-1.5, 1.7, step=0.2), fontsize=15)
plt.yticks(np.arange(-800, 1600, step=100), fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()
# plt.savefig(f'./results/images/obs_scatter.png', bbox_inches='tight')


fig = plt.figure(figsize=(10, 9))
plt.scatter(data_cl_obs['ma_acceleration'], data_cl_obs['msum_roc'], c=cmap(dataset_obs['labels']), alpha=0.7)
plt.xlabel('MA of acceleration', fontsize=15)
plt.ylabel('MS of ROC', fontsize=15)
plt.xticks(np.arange(-1.5, 1.7, step=0.2), fontsize=15)
plt.yticks(np.arange(-800, 1600, step=100), fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()
# plt.savefig(f'./results/images/obs_scatter_{nc}.png', bbox_inches='tight')


###############
print('DBI plot')

eval_obs = pd.read_csv(f'./results/observations/measures.csv', index_col=0)
eval_time = pd.read_csv(f'./results/time/measures.csv', index_col=0)

seconds = 10 * 60
obs_d = eval_obs.loc[:, str(10)].apply(lambda x: literal_eval(x)[0])
time_d = eval_time.loc[:, str(seconds)].apply(lambda x: literal_eval(x)[0])
index_d = eval_obs.index

fig = plt.figure(figsize=(10, 7))
plt.plot(index_d, time_d, marker="p", linestyle="-", linewidth=2,
         markersize=7, label='time-based')
plt.plot(index_d, obs_d, marker="p", linestyle="-", linewidth=2,
         markersize=7, label='message-based')
# plt.plot(index_d, np.repeat(0.5, len(obs_d)), color='black', linestyle="-", linewidth=2, label='threshold')
plt.ylabel('DBI', fontsize=15)
plt.xlabel('Number of clusters', fontsize=15)
plt.legend(fontsize=20)
plt.xticks(index_d, index_d, fontsize=15)
plt.yticks(np.arange(0.38, 0.55, step=0.02), fontsize=15)
plt.tight_layout()
plt.show()

