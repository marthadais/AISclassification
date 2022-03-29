import pandas as pd
from sklearn.cluster import KMeans
from preprocessing.clean_trajectories import Trajectories
from datetime import datetime
from preprocessing import features as fts
import os

print('Starting')
# https://coast.noaa.gov/data/marinecadastre/ais/VesselTypeCodes2018.pdf
#FISHING
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
    data_cl_all = fts.get_features(data_cl_all, n_dirs=n_dirs, win=window)
    data_cl_all.to_csv(features_path, index=False)
else:
    data_cl_all = pd.read_csv(features_path)

# select features to run the clustering
# data_cl = data_cl.drop(columns=['trajectory'])
# cols = ['ma_sog', 'ms_sog', 'ma_cog', 'ms_cog']
# cols = ['ma_acceleration', 'ms_acceleration', 'ma_roc', 'ms_roc']
# cols = ['ma_acceleration', 'ms_acceleration', 'ma_cog', 'ms_cog']
# cols = ['acceleration', 'ma_roc', 'ms_roc']
# cols = ['ma_sog', 'ms_sog', 'ma_roc', 'ms_roc']
# cols = ['ma_sog', 'ms_sog'] + list(range(n_dirs))
# cols = ['ma_acceleration', 'ms_acceleration'] + list(range(n_dirs))
# cols = ['ma_acceleration'] + list(range(n_dirs))
# cols = ['ma_sog'] + list(range(n_dirs))
# cols = ['acceleration'] + list(range(n_dirs))
# cols = ['sog'] + list(range(n_dirs))
# cols = ['ma_sog', 'ms_sog', 'ma_roc', 'ms_roc'] + list(range(n_dirs))
cols = ['ma_sog', 'ms_sog', 'ma_roc', 'ms_roc']
data_cl = data_cl_all[cols]

print('Clustering')
# Clustering
nc = 10
model = KMeans(nc).fit(data_cl)
labels = model.labels_
# metrics.silhouette_score(data_cl, labels)
data_cl['label'] = labels
data_cl_all['labels'] = labels

print('SOG values')
for c in range(nc):
    print(f"{c} = {data_cl_all[data_cl_all['labels']==c]['sog'].mean()}, {data_cl_all[data_cl_all['labels']==c]['sog'].std()}")
print('Rose values')
for c in range(nc):
    print(f"{c} = {data_cl_all[data_cl_all['labels']==c]['cog'].mean()}, {data_cl_all[data_cl_all['labels']==c]['cog'].std()}")

#Saving
# index of a few trajectories to quickly evaluate
test = data_cl_all[data_cl_all['trajectory'].isin([213, 117, 145, 26, 11])]
test.to_csv(f'fishing_5_{nc}_{window}_{cols}.csv', index=False)
data_cl_all.to_csv(f'fishing_{nc}_{window}_{cols}.csv', index=False)

