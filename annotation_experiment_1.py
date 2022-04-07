import pandas as pd
from sklearn.cluster import KMeans
from preprocessing import features as fts
import os

win = 10 # fixed number of observations
nc = 10 # number of clusters

print('Reading Dataset')
data_file = f'./data/preprocessed/DCAIS_[30_ 1001_ 1002]_None-mmsi_region_[46_ 51_ -130_ -122.5]_01-04_to_30-06_trips.csv'
dataset = pd.read_csv(data_file, parse_dates=['time'], low_memory=False)
dataset['time'] = dataset['time'].astype('datetime64[ns]')
dataset = dataset.sort_values(by=['trajectory', "time"])

print('Getting Features')
features_path = f'./data/features_window_{win}.csv'
if not os.path.exists(features_path):
    # features_all = fts.get_all_features(dataset, n_dirs=n_dirs, win=window, eps=seconds)
    features = fts.get_features(dataset, win=win)
    features.to_csv(features_path, index=False)
else:
    features = pd.read_csv(features_path)
data_cl = features[['ma_sog', 'msum_roc']]

print('Clustering')
# Clustering
model = KMeans(nc).fit(data_cl)
labels = model.labels_

print('Pos-processing')
labels2 = fts.posprocessing(labels)
data_cl['labels'] = labels2
dataset['labels'] = labels2

# Saving
# index of a few trajectories to quickly evaluate
test = dataset[dataset['trajectory'].isin([213, 117, 145, 26, 11])]
test.to_csv(f'1-fishing_5_{nc}_{win}.csv', index=False)
dataset.to_csv(f'1-fishing_{nc}_{win}.csv', index=False)
