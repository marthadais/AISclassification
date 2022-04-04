import pandas as pd
import numpy as np
from datetime import timedelta
from itertools import groupby


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
    diff_x[diff_x > 180] = 360 - diff_x[diff_x > 180]
    diff_x[diff_x < -180] = 360 + diff_x[diff_x < 180]
    return diff_x


def MA_MS_simple(data, col='sog', window=30, stats='mean'):
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


def MA_MS_timestamp(data, col='sog', epsilon=180, stats='mean', verbose=True):
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

    MA_data = MA_MS_timestamp(features, col='sog', epsilon=eps)
    features = pd.concat([features, MA_data], axis=1)
    MA_data = MA_MS_timestamp(features, col='cog', epsilon=eps)
    features = pd.concat([features, MA_data], axis=1)
    MA_data = MA_MS_timestamp(features, col='acceleration', epsilon=eps)
    features = pd.concat([features, MA_data], axis=1)
    MA_data = MA_MS_timestamp(features, col='roc', epsilon=eps)
    features = pd.concat([features, MA_data], axis=1)

    MA_data = MA_MS_simple(features, col='sog', window=win)
    features = pd.concat([features, MA_data], axis=1)
    MA_data = MA_MS_simple(features, col='cog', window=win)
    features = pd.concat([features, MA_data], axis=1)
    MA_data = MA_MS_simple(features, col='acceleration', window=win)
    features = pd.concat([features, MA_data], axis=1)
    MA_data = MA_MS_simple(features, col='roc', window=win)
    features = pd.concat([features, MA_data], axis=1)

    return features


def get_features_time(data, eps=180):
    features = remove_short_trajectories(data)
    features = features[['trajectory', 'time', 'sog', 'cog']]

    features['roc'] = diff_cog(features['cog'])
    features = features.fillna(0)
    MA_data = MA_MS_timestamp(features, col='sog', epsilon=eps, stats='mean')
    features = pd.concat([features, MA_data], axis=1)
    MA_data = MA_MS_timestamp(features, col='roc', epsilon=eps, stats='sum')
    features = pd.concat([features, MA_data], axis=1)

    return features


def get_features(data, win=10):
    features = remove_short_trajectories(data)
    features = features[['trajectory', 'time', 'sog', 'cog']]

    features['roc'] = diff_cog(features['cog'])
    features = features.fillna(0)
    MA_data = MA_MS_simple(features, col='sog', window=win)
    features = pd.concat([features, MA_data], axis=1)
    MA_data = MA_MS_simple(features, col='roc', window=win, stats='sum')
    features = pd.concat([features, MA_data], axis=1)

    return features


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