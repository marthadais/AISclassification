import pandas as pd
from sklearn.cluster import KMeans
from preprocessing.clean_trajectories import Trajectories
from datetime import datetime
from preprocessing import features as fts
import os

print('Starting')
# https://coast.noaa.gov/data/marinecadastre/ais/VesselTypeCodes2018.pdf
# FISHING
vessel_type = [30, 1001, 1002]
start_day = datetime(2020, 4, 1)
end_day = datetime(2020, 6, 30)
# Attributes
dim_set = ['lat', 'lon']
# Juan de Fuca Strait
region_limits = [46, 51, -130, -122.5]

# Creating dataset
print('Dataset')
dataset = Trajectories(n_samples=None, vessel_type=vessel_type, time_period=(start_day, end_day),
                       region=region_limits)

data_cl_all = dataset.get_dataset()

window = 5
n_dirs = 16
features_path = f'./data/DCAIS_{vessel_type}_-mmsi_region_{region_limits}_{start_day.day:02d}-{start_day.month:02d}_to_{end_day.day:02d}-{end_day.month:02d}_features.csv'
if not os.path.exists(features_path):
    features_all = fts.get_features(data_cl_all, n_dirs=n_dirs, win=window)
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
