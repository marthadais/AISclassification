import pandas as pd
from sklearn.cluster import KMeans
from preprocessing import features as fts
import os
from sklearn.metrics import davies_bouldin_score

folder = './results/observations_final/'
if not os.path.exists(folder):
    os.makedirs(folder)
    os.makedirs(f'{folder}/data')
win = 10 # fixed number of observations
nc = 10 # number of clusters

print('Reading Dataset')
data_file = f'./data/preprocessed/DCAIS_[30]_None-mmsi_region_[46, 51, -130, -122.5]_01-04_to_30-06_trips.csv'
dataset = pd.read_csv(data_file, parse_dates=['time'], low_memory=False)
dataset['time'] = dataset['time'].astype('datetime64[ns]')
dataset = dataset.sort_values(by=['trajectory', "time"])
dataset['control'] = dataset['status'].copy()
dataset.loc[dataset[dataset['status'] == 7].index, 'control'] = 1
dataset.loc[dataset[dataset['status'] != 7].index, 'control'] = 0

features_path = f'{folder}/data/features_window_{win}.csv'
if not os.path.exists(features_path):
    print('Getting Features')
    features = fts.get_features(dataset, win=win)
    features.to_csv(features_path, index=False)
else:
    print('Reading features file')
    features = pd.read_csv(features_path)
data_cl = features[['ma_acceleration', 'msum_roc']]

file_name = f'{folder}/fishing_{nc}_{win}.csv'
if not os.path.exists(file_name):
    # Clustering
    print('Clustering')
    model = KMeans(nc).fit(data_cl)
    labels = model.labels_
    data_cl.loc['labels'] = labels
    dataset.loc['labels'] = labels

    print('Pos-processing')
    labels2 = fts.posprocessing_2(dataset, min_points=5)['labels']
    data_cl.loc['labels_pos'] = labels2
    dataset.loc['labels_pos'] = labels2

    # Saving
    # index of a few trajectories to quickly evaluate
    trajs = dataset[dataset['status'] == 7]['trajectory'].unique()
    sample_fishing = dataset[dataset['trajectory'].isin(trajs)]
    sample_fishing.to_csv(f'{folder}/sample_fishing_{nc}_{win}.csv', index=False)
    dataset.to_csv(file_name, index=False)
else:
    print('Reading clustering files')
    dataset = pd.read_csv(file_name)
    sample_fishing = pd.read_csv(f'{folder}/sample_fishing_{nc}_{win}.csv')

print('Computing metrics')
DBI = davies_bouldin_score(features[['ma_acceleration', 'msum_roc']], dataset['labels'])
print(f'DBI data = {DBI}')
DBI = davies_bouldin_score(features[['ma_acceleration', 'msum_roc']], dataset['labels_pos'])
print(f'DBI pos data = {DBI}')




