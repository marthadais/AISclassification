import pandas as pd
import numpy as np
from datetime import timedelta
from itertools import groupby
from haversine import haversine


def remove_short_trajectories(data, n_obs=100):
    # remove trajectories with less than 100 observations
    obs_per_mmsi = data.groupby(data['mmsi'], as_index=False).size()
    ids_to_keep = obs_per_mmsi['mmsi'][obs_per_mmsi['size'] >= n_obs]
    return data[data['mmsi'].isin(ids_to_keep)]


# wind rose discretization
def cog_windrose(x, n_dirs=16):
    # dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    dirs = list(range(n_dirs))
    idx = np.round(np.array(x) / (360. / len(dirs))) % len(dirs)
    idx = idx.astype(int).tolist()
    return np.array(dirs)[idx]


def diff_cog(x):
    diff_x = x.diff()
    diff_x[diff_x > 180] = diff_x[diff_x > 180] - 360
    diff_x[diff_x < -180] = diff_x[diff_x < -180] + 360
    return diff_x


def MA_MS_analyze(data, verbose=True):
    mean_time_all = []
    mean_dist_all = []
    for traj_id in data['trajectory'].unique():
        if verbose:
            print(f'\t running trajectory {traj_id}')
        mean_window_traj = []
        trajectory_time = data[data['trajectory'] == traj_id]['time']
        trajectory_lat = data[data['trajectory'] == traj_id]['lat']
        trajectory_lat = trajectory_lat.reset_index(drop=True)
        trajectory_lon = data[data['trajectory'] == traj_id]['lon']
        trajectory_lon = trajectory_lon.reset_index(drop=True)
        delta_T = trajectory_time.diff()
        mean_time = []
        mean_dist = []

        for obs in range(1, trajectory_time.shape[0]-12):
            # get start point
            step = 0
            time_in_w = delta_T.iloc[obs]
            pointB = (trajectory_lat[obs], trajectory_lon[obs])
            distance = 0
            while (step <= 10):
                step = step + 1
                time_in_w = time_in_w + delta_T.iloc[obs+step]
                pointA = pointB
                pointB = (trajectory_lat[obs + step], trajectory_lon[obs + step])
                distance = distance + haversine(pointA, pointB)

            mean_time.append(time_in_w)
            mean_dist.append(distance)
        print(np.array(mean_time).mean())
        print(np.array(mean_dist).mean())

        mean_time_all.append(np.array(mean_time).mean())
        mean_dist_all.append(np.array(mean_dist).mean())

    if verbose:
        print(f'\t time window: {np.array(mean_time_all).mean()}, {np.array(mean_time_all).std()}')
        print(f'\t dist window: {np.array(mean_dist_all).mean()}, {np.array(mean_dist_all).std()}')


def MA_MS_simple(data, window=10):
    data_agg_sum = pd.DataFrame()
    data_agg_ma = pd.DataFrame()

    for traj_id in data['trajectory'].unique():
        trajectory_roc = data[data['trajectory'] == traj_id]['roc']
        trajectory_acc = data[data['trajectory'] == traj_id]['acceleration']
        # padding
        aux = pd.DataFrame(np.repeat(0, np.floor(window / 2)))
        aux2 = aux
        if window % 2 == 0:
            aux2 = aux2[0:-1]
        trajectory_roc = pd.concat([aux, trajectory_roc, aux2])
        trajectory_roc.reset_index(drop=True)
        trajectory_acc = pd.concat([aux, trajectory_acc, aux2])
        trajectory_acc.reset_index(drop=True)

        # Moving sum
        traj_roll_sum = pd.DataFrame()
        traj_roll_sum[f'msum_roc'] = trajectory_roc.rolling(window, center=True).sum()
        traj_roll_sum = traj_roll_sum.dropna()
        if traj_roll_sum.shape[0] == 0:
            print(f'Trajectory {traj_id} has not enough observations for window size {window}!')

        # Moving average
        traj_roll_ma = pd.DataFrame()
        traj_roll_ma[f'ma_acceleration'] = trajectory_acc.rolling(window, center=True).mean()
        traj_roll_ma = traj_roll_ma.dropna()
        if traj_roll_ma.shape[0] == 0:
            print(f'Trajectory {traj_id} has not enough observations for window size {window}!')
        data_agg_sum = pd.concat([data_agg_sum, traj_roll_sum], axis=0)
        data_agg_ma = pd.concat([data_agg_ma, traj_roll_ma], axis=0)

    data_agg_sum = data_agg_sum.reset_index(drop=True)
    data_agg_ma = data_agg_ma.reset_index(drop=True)
    data_agg = pd.concat([data_agg_sum, data_agg_ma], axis=1)

    return data_agg


def MA_MS_simple_2(data, col='sog', window=30, stats='mean'):
    data_agg = pd.DataFrame()
    for traj_id in data['trajectory'].unique():
        trajectory = data[data['trajectory'] == traj_id][col]
        # padding
        aux = pd.DataFrame(np.repeat(0, np.floor(window / 2)))
        aux2 = aux
        if window % 2 == 0:
            aux2 = aux2[0:-1]
        trajectory = pd.concat([aux, trajectory, aux2])
        trajectory.reset_index(drop=True)

        # Moving average
        traj_roll = pd.DataFrame()
        if stats == 'std':
            traj_roll[f'ms_{col}'] = trajectory.rolling(window, center=True).std()
        elif stats == 'sum':
            traj_roll[f'msum_{col}'] = trajectory.rolling(window, center=True).sum()
        else:
            traj_roll[f'ma_{col}'] = trajectory.rolling(window, center=True).mean()
        traj_roll = traj_roll.dropna()
        if traj_roll.shape[0] == 0:
            print(f'Trajectory {traj_id} has not enough observations for window size {window}!')
        data_agg = pd.concat([data_agg, traj_roll], axis=0)

    data_agg = data_agg.reset_index(drop=True)
    return data_agg


def MA_MS_timestamp(data, win=180, verbose=True):
    data_agg_roc = pd.DataFrame()
    data_agg_acc = pd.DataFrame()
    mean_window = []

    for traj_id in data['trajectory'].unique():
        if verbose:
            print(f'\t running trajectory {traj_id}')
        mean_window_traj = []
        trajectory_roc = data[data['trajectory'] == traj_id]['roc']
        trajectory_acc = data[data['trajectory'] == traj_id]['acceleration']
        trajectory_time = data[data['trajectory'] == traj_id]['time']
        delta_T = trajectory_time.diff()

        traj_roll_sum = []
        traj_roll_ma = []
        for obs in range(trajectory_roc.shape[0]):
            # get start point
            step = 0
            time_in_w = delta_T.iloc[obs]
            while (time_in_w.seconds <= win//2) and (obs-(step+1) >= 0):
                step = step + 1
                time_in_w = time_in_w + delta_T.iloc[obs-step]
            ini = obs - (step - 1)
            if step == 0:
                ini = obs
            # get end point
            step = 0
            time_in_w = timedelta(0)
            while (time_in_w.seconds <= win // 2) and (obs+(step+1) < trajectory_roc.shape[0]):
                step = step + 1
                time_in_w = time_in_w + delta_T.iloc[obs+step]
            end = obs + (step - 1)
            if step == 0:
                end = obs
            if np.isnan(trajectory_roc.iloc[ini:end].mean()):
                end = end + 1
            traj_roll_sum.append(trajectory_roc.iloc[ini:end].sum())
            traj_roll_ma.append(trajectory_acc.iloc[ini:end].mean())
            mean_window_traj.append(step)

        trajectory_roll_sum = pd.DataFrame()
        trajectory_roll_sum[f'msum_t_roc'] = pd.DataFrame(traj_roll_sum)
        data_agg_roc = pd.concat([data_agg_roc, trajectory_roll_sum], axis=0)

        trajectory_roll_ma = pd.DataFrame()
        trajectory_roll_ma[f'ma_t_acceleration'] = pd.DataFrame(traj_roll_ma)
        data_agg_acc = pd.concat([data_agg_acc, trajectory_roll_ma], axis=0)
        mean_window.append(np.array(mean_window_traj).mean())

    data_agg_roc = data_agg_roc.fillna(0)
    data_agg_acc = data_agg_acc.fillna(0)

    if verbose:
        print(f'\t time window: {np.array(mean_window).mean()}, {np.array(mean_window).std()}')
    data_agg_roc = data_agg_roc.reset_index(drop=True)
    data_agg_acc = data_agg_acc.reset_index(drop=True)
    data_agg = pd.concat([data_agg_roc, data_agg_acc], axis=1)

    return data_agg


def MA_MS_timestamp2(data, col='sog', epsilon=180, stats='mean', verbose=True):
    data_agg = pd.DataFrame()
    mean_window = []
    if verbose:
        print(f'Running for {col}')
    for traj_id in data['trajectory'].unique():
        if verbose:
            print(f'\t running trajectory {traj_id}')
        mean_window_traj = []
        trajectory = data[data['trajectory'] == traj_id][col]
        trajectory_time = data[data['trajectory'] == traj_id]['time']
        delta_T = trajectory_time.diff()

        traj_roll = []
        for obs in range(trajectory.shape[0]):
            # get start point
            step = 0
            time_in_w = delta_T.iloc[obs]
            while (time_in_w.seconds <= epsilon//2) and (obs-(step+1) >= 0):
                step = step + 1
                time_in_w = time_in_w + delta_T.iloc[obs-step]
            ini = obs - (step - 1)
            if step == 0:
                ini = obs
            # get end point
            step = 0
            time_in_w = timedelta(0)
            while (time_in_w.seconds <= epsilon // 2) and (obs+(step+1) < trajectory.shape[0]):
                step = step + 1
                time_in_w = time_in_w + delta_T.iloc[obs+step]
            end = obs + (step - 1)
            if step == 0:
                end = obs
            if np.isnan(trajectory.iloc[ini:end].mean()):
                end = end + 1
            if stats == 'std':
                traj_roll.append(trajectory.iloc[ini:end].std())
            elif stats == 'sum':
                traj_roll.append(trajectory.iloc[ini:end].sum())
            else:
                traj_roll.append(trajectory.iloc[ini:end].mean())
            mean_window_traj.append(step)

        trajectory_roll = pd.DataFrame()
        if stats == 'std':
            trajectory_roll[f'ms_t_{col}'] = pd.DataFrame(traj_roll)
        elif stats == 'sum':
            trajectory_roll[f'msum_t_{col}'] = pd.DataFrame(traj_roll)
        else:
            trajectory_roll[f'ma_t_{col}'] = pd.DataFrame(traj_roll)
        data_agg = pd.concat([data_agg, trajectory_roll], axis=0)
        mean_window.append(np.array(mean_window_traj).mean())

    data_agg = data_agg.fillna(0)
    if verbose:
        print(f'\t time window: {np.array(mean_window).mean()}, {np.array(mean_window).std()}')
    data_agg = data_agg.reset_index(drop=True)
    return data_agg


def MA_MS_distance(data, win=180, verbose=True):
    data_agg_roc = pd.DataFrame()
    data_agg_acc = pd.DataFrame()
    mean_window = []

    for traj_id in data['trajectory'].unique():
        if verbose:
            print(f'\t running trajectory {traj_id}')
        mean_window_traj = []
        trajectory_roc = data[data['trajectory'] == traj_id]['roc']
        trajectory_acc = data[data['trajectory'] == traj_id]['acceleration']
        trajectory_lat = data[data['trajectory'] == traj_id]['lat']
        trajectory_lat = trajectory_lat.reset_index(drop=True)
        trajectory_lon = data[data['trajectory'] == traj_id]['lon']
        trajectory_lon = trajectory_lon.reset_index(drop=True)

        traj_roll_roc = []
        traj_roll_acc = []
        for obs in range(trajectory_roc.shape[0]):
            step = 0
            pointB = (trajectory_lat[obs], trajectory_lon[obs])
            distance = 0
            while (distance <= win // 2) and (obs - (step + 1) >= 0):
                step = step + 1
                pointA = pointB
                pointB = (trajectory_lat[obs-step], trajectory_lon[obs-step])
                distance = distance + haversine(pointA, pointB)
            ini = obs - (step - 1)
            if step == 0:
                ini = obs
            # get end point
            step = 0
            pointB = (trajectory_lat[obs], trajectory_lon[obs])
            distance = 0
            while (distance <= win // 2) and (obs + (step + 1) < trajectory_roc.shape[0]):
                step = step + 1
                pointA = pointB
                pointB = (trajectory_lat[obs + step], trajectory_lon[obs + step])
                distance = distance + haversine(pointA, pointB)
            end = obs + (step - 1)
            if step == 0:
                end = obs
            if np.isnan(trajectory_roc.iloc[ini:end].mean()):
                end = end + 1
            if np.isnan(trajectory_acc.iloc[ini:end].mean()):
                end = end + 1

            traj_roll_roc.append(trajectory_roc.iloc[ini:end].sum())
            traj_roll_acc.append(trajectory_acc.iloc[ini:end].mean())
            mean_window_traj.append(step)

        trajectory_roll_roc = pd.DataFrame()
        trajectory_roll_roc[f'msum_d_roc'] = pd.DataFrame(traj_roll_roc)
        trajectory_roll_acc = pd.DataFrame()
        trajectory_roll_acc[f'ma_d_acceleration'] = pd.DataFrame(traj_roll_acc)

        data_agg_roc = pd.concat([data_agg_roc, trajectory_roll_roc], axis=0)
        data_agg_acc = pd.concat([data_agg_acc, trajectory_roll_acc], axis=0)
        mean_window.append(np.array(mean_window_traj).mean())

    data_agg_roc = data_agg_roc.fillna(0)
    data_agg_acc = data_agg_acc.fillna(0)
    if verbose:
        print(f'\t distance window: {np.array(mean_window).mean()}, {np.array(mean_window).std()}')
    data_agg_roc = data_agg_roc.reset_index(drop=True)
    data_agg_acc = data_agg_acc.reset_index(drop=True)
    data_agg = pd.concat([data_agg_roc, data_agg_acc], axis=1)

    return data_agg


def MA_MS_distance2(data, col='sog', epsilon=180, stats='mean', verbose=True):
    data_agg = pd.DataFrame()
    mean_window = []
    if verbose:
        print(f'Running for {col}')
    for traj_id in data['trajectory'].unique():
        if verbose:
            print(f'\t running trajectory {traj_id}')
        mean_window_traj = []
        trajectory = data[data['trajectory'] == traj_id][col]
        trajectory_lat = data[data['trajectory'] == traj_id]['lat']
        trajectory_lat = trajectory_lat.reset_index(drop=True)
        trajectory_lon = data[data['trajectory'] == traj_id]['lon']
        trajectory_lon = trajectory_lon.reset_index(drop=True)

        traj_roll = []
        for obs in range(trajectory.shape[0]):
            step = 0
            pointB = (trajectory_lat[obs], trajectory_lon[obs])
            distance = 0
            while (distance <= epsilon // 2) and (obs - (step + 1) >= 0):
                step = step + 1
                pointA = pointB
                pointB = (trajectory_lat[obs-step], trajectory_lon[obs-step])
                distance = distance + haversine(pointA, pointB)
            ini = obs - (step - 1)
            if step == 0:
                ini = obs
            # get end point
            step = 0
            pointB = (trajectory_lat[obs], trajectory_lon[obs])
            distance = 0
            while (distance <= epsilon // 2) and (obs + (step + 1) < trajectory.shape[0]):
                step = step + 1
                pointA = pointB
                pointB = (trajectory_lat[obs + step], trajectory_lon[obs + step])
                distance = distance + haversine(pointA, pointB)
            end = obs + (step - 1)
            if step == 0:
                end = obs
            if np.isnan(trajectory.iloc[ini:end].mean()):
                end = end + 1
            if stats == 'std':
                traj_roll.append(trajectory.iloc[ini:end].std())
            elif stats == 'sum':
                traj_roll.append(trajectory.iloc[ini:end].sum())
            else:
                traj_roll.append(trajectory.iloc[ini:end].mean())
            mean_window_traj.append(step)

        trajectory_roll = pd.DataFrame()
        if stats == 'std':
            trajectory_roll[f'ms_d_{col}'] = pd.DataFrame(traj_roll)
        elif stats == 'sum':
            trajectory_roll[f'msum_d_{col}'] = pd.DataFrame(traj_roll)
        else:
            trajectory_roll[f'ma_d_{col}'] = pd.DataFrame(traj_roll)
        data_agg = pd.concat([data_agg, trajectory_roll], axis=0)
        mean_window.append(np.array(mean_window_traj).mean())

    data_agg = data_agg.fillna(0)
    if verbose:
        print(f'\t distance window: {np.array(mean_window).mean()}, {np.array(mean_window).std()}')
    data_agg = data_agg.reset_index(drop=True)
    return data_agg


def get_all_features(data, n_dirs=4, win=30, eps=180):
    features = remove_short_trajectories(data)
    features = features[['trajectory', 'time', 'sog', 'cog']]
    features['rose'] = cog_windrose(features['cog'], n_dirs=n_dirs)
    # encoding
    enc_rose = pd.get_dummies(features['rose'])
    features = pd.concat([features, enc_rose], axis=1)

    features['roc'] = diff_cog(features['cog'])
    features['acceleration'] = features['sog'].diff()/features['time'].diff().dt.seconds
    features = features.fillna(0)

    MA_data = MA_MS_timestamp2(features, col='sog', epsilon=eps)
    features = pd.concat([features, MA_data], axis=1)
    MA_data = MA_MS_timestamp2(features, col='cog', epsilon=eps)
    features = pd.concat([features, MA_data], axis=1)
    MA_data = MA_MS_timestamp2(features, col='acceleration', epsilon=eps)
    features = pd.concat([features, MA_data], axis=1)
    MA_data = MA_MS_timestamp2(features, col='roc', epsilon=eps)
    features = pd.concat([features, MA_data], axis=1)

    MA_data = MA_MS_simple_2(features, col='sog', window=win)
    features = pd.concat([features, MA_data], axis=1)
    MA_data = MA_MS_simple_2(features, col='cog', window=win)
    features = pd.concat([features, MA_data], axis=1)
    MA_data = MA_MS_simple_2(features, col='acceleration', window=win)
    features = pd.concat([features, MA_data], axis=1)
    MA_data = MA_MS_simple_2(features, col='roc', window=win)
    features = pd.concat([features, MA_data], axis=1)

    return features


def get_features_distance2(data, eps=10):
    features = remove_short_trajectories(data)
    features = features[['trajectory', 'time', 'lat', 'lon', 'sog', 'cog']]

    features['roc'] = diff_cog(features['cog'])
    features['acceleration'] = features['sog'].diff()/features['time'].diff().dt.seconds
    features = features.fillna(0)
    MA_data = MA_MS_distance2(features, col='sog', epsilon=eps, stats='mean')
    features = pd.concat([features, MA_data], axis=1)
    MA_data = MA_MS_distance2(features, col='roc', epsilon=eps, stats='sum')
    features = pd.concat([features, MA_data], axis=1)
    MA_data = MA_MS_distance2(features, col='acceleration', epsilon=eps, stats='mean')
    features = pd.concat([features, MA_data], axis=1)

    return features


def get_features_distance(data, win=10):
    # features = remove_short_trajectories(data)
    features = data[['trajectory', 'time', 'lat', 'lon', 'sog', 'cog']]

    features['roc'] = diff_cog(features['cog'])
    features['acceleration'] = features['sog'].diff()/features['time'].diff().dt.seconds
    features = features.fillna(0)
    MA_data = MA_MS_distance(features, win=win)
    features = pd.concat([features, MA_data], axis=1)

    return features


def get_features_time(data, win=180):
    # features = remove_short_trajectories(data)
    features = data[['trajectory', 'time', 'sog', 'cog']]

    features['roc'] = diff_cog(features['cog'])
    features['acceleration'] = features['sog'].diff()/features['time'].diff().dt.seconds
    features = features.fillna(0)
    MA_data = MA_MS_timestamp(features, win=win)
    features = pd.concat([features, MA_data], axis=1)

    return features


def get_features_time2(data, eps=180):
    features = remove_short_trajectories(data)
    features = features[['trajectory', 'time', 'sog', 'cog']]

    features['roc'] = diff_cog(features['cog'])
    features['acceleration'] = features['sog'].diff()/features['time'].diff().dt.seconds
    features = features.fillna(0)
    MA_data = MA_MS_timestamp2(features, col='sog', epsilon=eps, stats='mean')
    features = pd.concat([features, MA_data], axis=1)
    MA_data = MA_MS_timestamp2(features, col='roc', epsilon=eps, stats='sum')
    features = pd.concat([features, MA_data], axis=1)
    MA_data = MA_MS_timestamp2(features, col='acceleration', epsilon=eps, stats='mean')
    features = pd.concat([features, MA_data], axis=1)

    return features


def get_features_2(data, win=10):
    features = remove_short_trajectories(data)
    features = features[['trajectory', 'time', 'sog', 'cog']]

    features['roc'] = diff_cog(features['cog'])
    features['acceleration'] = features['sog'].diff() / features['time'].diff().dt.seconds
    features = features.fillna(0)
    MA_data = MA_MS_simple_2(features, col='sog', window=win)
    features = pd.concat([features, MA_data], axis=1)
    MA_data = MA_MS_simple_2(features, col='roc', window=win, stats='sum')
    features = pd.concat([features, MA_data], axis=1)
    MA_data = MA_MS_simple_2(features, col='acceleration', window=win)
    features = pd.concat([features, MA_data], axis=1)

    return features


def get_features(data, win=10):
    # features = remove_short_trajectories(data)
    features = data[['trajectory', 'time', 'sog', 'cog']]

    features['roc'] = diff_cog(features['cog'])
    features['acceleration'] = features['sog'].diff() / features['time'].diff().dt.seconds
    features = features.fillna(0)

    MA_data = MA_MS_simple(features, window=win)
    features = pd.concat([features, MA_data], axis=1)

    return features


def posprocessing(x, min_points=2, verbose=False, labels2=True):
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
        if len(s) <= min_points:
            if len(new_list) == 0:
                new_list = new_list + s
            else:
                new_list = new_list + [new_list[-1] for i in range(min_points)]
        else:
            new_list = new_list + s
        c = c+1

    return new_list


def posprocessing_2(data, min_points=1, verbose=False):
    x = data.copy()
    lbl = x['labels'].tolist()

    cl = max(set(lbl), key=lbl.count)
    idx = x[x['labels'] != cl].index
    x.loc[idx, 'labels'] = -1
    idx = x[x['labels'] == cl].index
    x.loc[idx, 'labels'] = 0
    x['labels'] = list(abs(x['labels']))

    for t in x['trajectory'].unique():
        if verbose:
            print(f'{t} of {len(x.trajectory.unique())}')
        lbl = x[x['trajectory'] == t]['labels'].tolist()
        new_list = []
        c = 0
        for k, g in groupby(lbl):
            s = list(g)
            if verbose:
                print(f'\t{c} of {len(lbl)}, {len(s)}')
            if len(s) <= min_points:
                if len(new_list) == 0:
                    new_list = new_list + s
                else:
                    new_list = new_list + [new_list[-1] for i in range(len(s))]
            else:
                new_list = new_list + s
            c = c + len(s)
        idx = x[x['trajectory'] == t].index
        x.loc[idx, 'labels'] = new_list
    return x