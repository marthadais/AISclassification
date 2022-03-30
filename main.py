import pandas as pd
from sklearn.cluster import KMeans
from preprocessing import features as fts
import os

print('Reading Dataset')
data_file = f'./data/preprocessed/DCAIS_[30_ 1001_ 1002]_None-mmsi_region_[46_ 51_ -130_ -122.5]_01-04_to_30-06_trips.csv'
dataset = pd.read_csv(data_file, parse_dates=['time'], low_memory=False)
dataset['time'] = dataset['time'].astype('datetime64[ns]')
dataset = dataset.sort_values(by=['trajectory', "time"])

window = 5
n_dirs = 16
seconds = 10*60
features_path = f'./data/features_win_{window}_rose_{n_dirs}_time_win_{seconds}.csv'
if not os.path.exists(features_path):
    features_all = fts.get_features(dataset, n_dirs=n_dirs, win=window, eps=seconds)
    features_all.to_csv(features_path, index=False)
else:
    features_all = pd.read_csv(features_path)

# select features to run the clustering
# features columns:'sog', 'cog', 'rose',
# 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, ([str(i) for i in range(n_dirs)])
# 'roc', 'acceleration', 'ma_t_sog', 'ms_t_sog', 'ma_t_cog', 'ms_t_cog',
# 'ma_t_acceleration', 'ms_t_acceleration', 'ma_t_roc', 'ms_t_roc',
# 'ma_sog', 'ms_sog', 'ma_cog', 'ms_cog',
# 'ma_acceleration', 'ms_acceleration', 'ma_roc', 'ms_roc'

#%%
cols = ['ma_t_sog'] + [str(i) for i in range(n_dirs)]
data_cl = features_all[cols]

print('Clustering')
# Clustering
nc = 10
model = KMeans(nc).fit(data_cl)
labels = model.labels_
# metrics.silhouette_score(data_cl, labels)
data_cl['label'] = labels
data_cl_all['labels'] = labels

# Saving
# index of a few trajectories to quickly evaluate
test = data_cl_all[data_cl_all['trajectory'].isin([213, 117, 145, 26, 11])]
test.to_csv(f'fishing_5_{nc}_{window}_{cols}.csv', index=False)
data_cl_all.to_csv(f'fishing_{nc}_{window}_{cols}.csv', index=False)
