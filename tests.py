import pandas as pd
from sklearn.cluster import KMeans
from preprocessing import features as fts
import os
from itertools import groupby
from collections import Counter


def posprocessing(x, min_points=1, verbose=False, labels2=True):
    """
    It applies a pos-processing on the clustering labels.
    It changes the label of the current observation to the previous one
     when the current observation and the next ones of same label have less
     than a minimum number
    It also converts lines as cluster 0 and curves as cluster 1
    :param x: labels sequence (array)
    :param min_points: minimum number in the sequence to change the label
    :param verbose: if True, it will print something
    :param labels2: if True, converts lines as cluster 0 and curves as cluster 1
    :return: the new list of labels
    """
    lbl = x.tolist()

    if labels2:
        cl = max(set(lbl), key=lbl.count)
        x[x != cl] = -1
        x[x == cl] = 0
        lbl = list(abs(x))

    new_list = []
    c=0
    for k, g in groupby(lbl):
        s = list(g)
        if verbose:
            print(f'{c} of {len(lbl)}')
        if len(s) == min_points:
            if len(new_list) == 0:
                new_list = new_list + [0]
            else:
                new_list = new_list + [new_list[-1] for i in range(min_points)]
        else:
            new_list = new_list + s
        c = c+1

    return new_list

print('Starting all Experiments...')
from preprocessing.clean_trajectories import Trajectories
from datetime import datetime
n_samples = None
vessel_type = [30, 50, 60, 37]
start_day = datetime(2020, 4, 1)
end_day = datetime(2020, 4, 30)
region_limits = [46,51,-130,-122.5]

vessel_type = [30]
start_day = datetime(2020, 4, 1)
end_day = datetime(2020, 9, 30)
region_limits = [46,51,-130,-122.5]

### Creating dataset
# dataset = Trajectories(n_samples=n_samples, vessel_type=vessel_type, time_period=(start_day, end_day), region=region_limits)


print('Reading Dataset')
# data_file = f'./data/preprocessed/DCAIS_[30_ 1001_ 1002]_None-mmsi_region_[46_ 51_ -130_ -122.5]_01-04_to_30-06_trips.csv'
# data_file = f'./data/preprocessed/DCAIS_[30, 50, 60, 37]_None-mmsi_region_[46, 51, -130, -122.5]_01-04_to_30-04_trips.csv'
data_file = f'./data/preprocessed/DCAIS_[30]_None-mmsi_region_[46, 51, -130, -122.5]_01-04_to_30-06_trips.csv'
dataset = pd.read_csv(data_file, parse_dates=['time'], low_memory=False)
dataset['time'] = dataset['time'].astype('datetime64[ns]')
dataset = dataset.sort_values(by=['trajectory', "time"])

dataset['control'] = dataset['status']
dataset.loc[dataset[dataset['status'] == 7].index, 'control'] = 1
dataset.loc[dataset[dataset['status'] != 7].index, 'control'] = 0

# seconds = 10*60 # window seconds
# features_path = f'./data/features_window_{seconds}.csv'
# if not os.path.exists(features_path):
#     # features_all = fts.get_all_features(dataset, n_dirs=n_dirs, win=window, eps=seconds)
#     features = fts.get_features_time(dataset, eps=seconds)
#     features.to_csv(features_path, index=False)
# else:
#     features = pd.read_csv(features_path)

seconds = 10*60
win = 10
features_path = f'./data/features_all_window_{win}_time_{seconds}.csv'
if not os.path.exists(features_path):
    # features_all = fts.get_all_features(dataset, n_dirs=n_dirs, win=window, eps=seconds)
    features = fts.get_features(dataset, win=win)
    features.to_csv(features_path, index=False)
else:
    features = pd.read_csv(features_path)

# select features to run the clustering
# features columns:'sog', 'cog', 'rose',
# 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, ([str(i) for i in range(n_dirs)])
# 'roc', 'acceleration', 'ma_t_sog', 'ms_t_sog', 'msum_t_sog', 'ma_t_cog', 'ms_t_cog', 'msum_t_cog',
# 'ma_t_acceleration', 'ms_t_acceleration', 'msum_t_acceleration', 'ma_t_roc', 'ms_t_roc', 'msum_t_roc',
# 'ma_sog', 'ms_sog', 'msum_sog', 'ma_cog', 'ms_cog', 'msum_cog',
# 'ma_acceleration', 'ms_acceleration', 'msum_acceleration', 'ma_roc', 'ms_roc', 'msum_roc'
# [str(i) for i in range(n_dirs)]

#%%
data_cl = features[['msum_t_acceleration', 'msum_t_roc']]

print('Clustering')
# Clustering
nc = 10
model = KMeans(nc).fit(data_cl)
labels = model.labels_

print('Pos-processing')
# labels2 = posprocessing(labels)
data_cl['labels'] = labels
dataset['labels'] = labels

print('Computing metrics')
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
comp_labels = normalized_mutual_info_score(dataset['control'], labels)
sl_data = silhouette_score(features[['msum_acceleration', 'msum_roc']], labels)
sl_control = silhouette_score(features[['msum_acceleration', 'msum_roc']], dataset['control'])


# Saving
# index of a few trajectories to quickly evaluate
test = dataset[dataset['trajectory'].isin([213, 117, 145, 26, 11])]
test.to_csv(f'fishing_5_{nc}_{win}.csv', index=False)
dataset.to_csv(f'fishing_{nc}_{win}.csv', index=False)

# from sklearn.metrics import normalized_mutual_info_score, davies_bouldin_score, calinski_harabasz_score
# comp_labels = normalized_mutual_info_score(sample_fishing['control'], sample_fishing['labels_pos'])
# print(f'MI = {comp_labels}')
# sl_data = silhouette_score(test_cl[['ma_sog', 'msum_roc']], test.csv['labels'])
# print(f'S data = {sl_data}')
# sl_control = silhouette_score(test_cl[['ma_sog', 'msum_roc']], test.csv['control'])
# print(f'S control = {sl_control}')

# CH = calinski_harabasz_score(features[['ma_sog', 'msum_roc']], dataset['labels'])
# print(f'CH data = {CH}')
# CH = calinski_harabasz_score(features[['ma_sog', 'msum_roc']], dataset['control'])
# print(f'CH control = {CH}')
# CH = calinski_harabasz_score(features[['ma_sog', 'msum_roc']], dataset['labels_pos'])
# print(f'CH pos data = {CH}')


# prev = pd.read_csv(f'{folder}/fishing_7_{win}.csv')
# for nc in range(3, nc_size):
#     file_name = f'{folder}/fishing_{nc}_{win}.csv'
#     dataset = pd.read_csv(file_name)
#     MI = normalized_mutual_info_score(prev['labels_pos'], dataset['labels_pos'])
#     print(f'MI 14 and {nc} = {MI}')

# We store the clusters
# clus0 = data_cl.loc[data_cl['labels'] == 0]
# clus1 = data_cl.loc[data_cl['labels'] == 1]
# cluster_list = [clus0.values, clus1.values]
# print(base.dunn(cluster_list))


# plt.title(f'{nc}')
# plt.scatter(features['ma_sog'], features['msum_roc'], c=dataset['labels'])
# plt.show()