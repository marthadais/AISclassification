import pandas as pd
from sklearn.cluster import KMeans
from preprocessing import features as fts
import os
from sklearn.metrics import davies_bouldin_score, silhouette_score
from preprocessing.dunn_index import bootstrap_sampling
import numpy as np

sec_size = 21 # fixed number of observations
nc_size = 21 # number of clusters to be tested
eval = {}

print('Reading Dataset')
data_file = f'./data/preprocessed/DCAIS_[30]_None-mmsi_region_[46, 51, -130, -122.5]_01-04_to_30-06_trips.csv'
dataset_raw = pd.read_csv(data_file, parse_dates=['time'], low_memory=False)
dataset_raw['time'] = dataset_raw['time'].astype('datetime64[ns]')
dataset_raw = dataset_raw.sort_values(by=['trajectory', "time"])
dataset_raw['control'] = dataset_raw['status'].copy()
dataset_raw.loc[dataset_raw[dataset_raw['status'] == 7].index, 'control'] = 1
dataset_raw.loc[dataset_raw[dataset_raw['status'] != 7].index, 'control'] = 0

for seconds in (np.array(range(2, sec_size)) * 60):
    dataset = dataset_raw.copy()
    folder = f'./results/time_{seconds}/'
    if not os.path.exists(folder):
        os.makedirs(folder)
        os.makedirs(f'{folder}/data')

    features_path = f'{folder}/data/features_window_time_{seconds}.csv'
    print(features_path)
    if not os.path.exists(features_path):
        print('Getting Features')
        # features_all = fts.get_all_features(dataset, n_dirs=n_dirs, win=window, eps=seconds)
        features = fts.get_features_time(dataset, eps=seconds)
        features.to_csv(features_path, index=False)
    else:
        print('Reading features file')
        features = pd.read_csv(features_path)
    data_cl = features[['ma_t_acceleration', 'msum_t_roc']]

    eval[seconds] = {}
    for nc in range(2, nc_size):
        file_name = f'{folder}/fishing_{nc}_{seconds}.csv'
        if not os.path.exists(file_name):
            # Clustering
            print(f'Clustering - {nc}')
            model = KMeans(nc).fit(data_cl)
            labels = model.labels_
            data_cl['labels'] = labels
            dataset['labels'] = labels

            print('Pos-processing')
            # labels2 = fts.posprocessing(labels, min_points=seconds)
            labels2 = fts.posprocessing_2(dataset, min_points=5)['labels']
            data_cl['labels_pos'] = labels2
            dataset['labels_pos'] = labels2

            # Saving
            # index of a few trajectories to quickly evaluate
            trajs = dataset[dataset['status'] == 7]['trajectory'].unique()
            sample_fishing = dataset[dataset['trajectory'].isin(trajs)]
            sample_fishing.to_csv(f'{folder}/sample_fishing_{nc}_{seconds}.csv', index=False)
            dataset.to_csv(file_name, index=False)
        else:
            print('Reading clustering files')
            dataset = pd.read_csv(file_name)
            sample_fishing = pd.read_csv(f'{folder}/sample_fishing_{nc}_{seconds}.csv')
            test = dataset[dataset['trajectory'].isin([213, 117, 145, 26, 11])]
            if not os.path.exists(f'{folder}/test/'):
                os.makedirs(f'{folder}/test/')
            test.to_csv(f'{folder}/test/test_{nc}_{seconds}.csv', index=False)

        print('Computing metrics')
        DBI_l = davies_bouldin_score(features[['ma_t_acceleration', 'msum_t_roc']], dataset['labels'])
        print(f'DBI data = {DBI_l}')
        DBI_lp = davies_bouldin_score(features[['ma_t_acceleration', 'msum_t_roc']], dataset['labels_pos'])
        print(f'DBI pos data = {DBI_lp}')

        # dunn, sil = bootstrap_sampling(features[['ma_t_acceleration', 'msum_t_roc']], dataset['labels'])
        # dunn_lp, sil_lp = bootstrap_sampling(features[['ma_t_acceleration', 'msum_t_roc']], dataset['labels_pos'])

        eval[seconds][nc] = [DBI_l, DBI_lp]
        # eval[seconds][nc] = [DBI_l, DBI_lp, dunn, dunn_lp, sil, sil_lp]
        eval2 = pd.DataFrame.from_dict(eval)
        eval2.to_csv(f'./results/time/measures.csv')

eval2 = pd.DataFrame.from_dict(eval)
eval2.to_csv(f'./results/time/measures.csv')




