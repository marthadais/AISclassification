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

print('Reading Dataset')
data_file = f'./data/preprocessed/DCAIS_[30_ 1001_ 1002]_None-mmsi_region_[46_ 51_ -130_ -122.5]_01-04_to_30-06_trips.csv'
dataset = pd.read_csv(data_file, parse_dates=['time'], low_memory=False)
dataset['time'] = dataset['time'].astype('datetime64[ns]')
dataset = dataset.sort_values(by=['trajectory', "time"])

# seconds = 10*60 # window seconds
# features_path = f'./data/features_window_{seconds}.csv'
# if not os.path.exists(features_path):
#     # features_all = fts.get_all_features(dataset, n_dirs=n_dirs, win=window, eps=seconds)
#     features = fts.get_features_time(dataset, eps=seconds)
#     features.to_csv(features_path, index=False)
# else:
#     features = pd.read_csv(features_path)


win = 10 # fixed window size
features_path = f'./data/features_window_{win}.csv'
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
data_cl = features[['ma_sog', 'msum_roc']]

print('Clustering')
# Clustering
nc = 10
model = KMeans(nc).fit(data_cl)
labels = model.labels_

print('Pos-processing')
labels2 = posprocessing(labels)
data_cl['labels'] = labels2
dataset['labels'] = labels2


# Saving
# index of a few trajectories to quickly evaluate
test = dataset[dataset['trajectory'].isin([213, 117, 145, 26, 11])]
test.to_csv(f'fishing_5_{nc}_{win}.csv', index=False)
dataset.to_csv(f'fishing_{nc}_{win}.csv', index=False)
