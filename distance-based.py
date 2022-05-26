import pandas as pd
from sklearn.cluster import KMeans
from preprocessing import features as fts
import os
from sklearn.metrics import davies_bouldin_score

folder = './results/distance_final/'
if not os.path.exists(folder):
    os.makedirs(folder)
    os.makedirs(f'{folder}/data')

meters = 5000/1000 # dist-based window
nc = 12 # number of clusters

print('Reading Dataset')
data_file = f'./data/preprocessed/DCAIS_[30]_None-mmsi_region_[46, 51, -130, -122.5]_01-04_to_30-06_trips.csv'
dataset = pd.read_csv(data_file, parse_dates=['time'], low_memory=False)
dataset['time'] = dataset['time'].astype('datetime64[ns]')
dataset = dataset.sort_values(by=['trajectory', "time"])
dataset['control'] = dataset['status'].copy()
dataset.loc[dataset[dataset['status'] == 7].index, 'control'] = 1
dataset.loc[dataset[dataset['status'] != 7].index, 'control'] = 0

print('Getting Features')
features_path = f'{folder}/data/features_window_dist_{meters}.csv'
if not os.path.exists(features_path):
    # features_all = fts.get_all_features(dataset, n_dirs=n_dirs, win=window, eps=seconds)
    features = fts.get_features_distance(dataset, eps=meters)
    features.to_csv(features_path, index=False)
else:
    features = pd.read_csv(features_path)
data_cl = features[['ma_d_acceleration', 'msum_d_roc']]

file_name = f'{folder}/fishing_{nc}_{meters}.csv'
if not os.path.exists(file_name):
    print('Clustering')
    # Clustering
    model = KMeans(nc).fit(data_cl)
    labels = model.labels_
    data_cl['labels'] = labels
    dataset['labels'] = labels

    print('Pos-processing')
    labels2 = fts.posprocessing_2(dataset, min_points=5)['labels']
    data_cl['labels_pos'] = labels2
    dataset['labels_pos'] = labels2

    # Saving
    # index of a few trajectories to quickly evaluate
    trajs = dataset[dataset['status']==7]['trajectory'].unique()
    sample_fishing = dataset[dataset['trajectory'].isin(trajs)]
    sample_fishing.to_csv(f'{folder}/sample_fishing_{nc}_{meters}.csv', index=False)
    dataset.to_csv(f'{folder}/fishing_{nc}_{meters}.csv', index=False)

    t_size = len(dataset['trajectory'].unique())
    idx_train = list(range(round(t_size * 0.7)))
    data_train = dataset[dataset['trajectory'].isin(idx_train)]
    data_train.to_csv(f'{folder}/train_fishing_{nc}_{meters}.csv', index=False)
    idx_test = list(range(round(t_size * 0.7), t_size))
    data_test = dataset[dataset['trajectory'].isin(idx_test)]
    data_test.to_csv(f'{folder}/test_fishing_{nc}_{meters}.csv', index=False)


else:
    print('Reading clustering files')
    dataset = pd.read_csv(file_name)
    sample_fishing = pd.read_csv(f'{folder}/sample_fishing_{nc}_{meters}.csv')

print('Computing metrics')
DBI = davies_bouldin_score(features[['ma_d_acceleration', 'msum_d_roc']], dataset['labels'])
print(f'DBI data = {DBI}')
DBI = davies_bouldin_score(features[['ma_d_acceleration', 'msum_d_roc']], dataset['labels_pos'])
print(f'DBI pos data = {DBI}')


