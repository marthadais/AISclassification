import pandas as pd
import numpy as np
from datetime import timedelta


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


def MA_MS_simple(data, col='sog', window=30):
    data_agg = pd.DataFrame()
    for traj_id in data['trajectory'].unique():
        trajectory = data[data['trajectory'] == traj_id][col]
        # padding
        aux = pd.DataFrame(np.repeat(0, np.floor(window / 2)))
        aux2 = aux
        if window % 2 == 0:
            aux2 = aux2[0:-1]
        trajectory = pd.concat([aux, trajectory, aux2])

        # Moving average
        traj_roll = pd.DataFrame()
        traj_roll[f'ma_{col}'] = trajectory.rolling(window, center=True).mean()
        traj_roll[f'ms_{col}'] = trajectory.rolling(window, center=True).std()
        traj_roll = traj_roll.dropna()
        if traj_roll.shape[0] == 0:
            print(f'Trajectory {traj_id} has not enough observations for window size {window}!')
        data_agg = pd.concat([data_agg, traj_roll], axis=0)

    data_agg = data_agg.reset_index(drop=True)
    return data_agg


def MA_MS_timestamp(data, col='sog', epsilon=180, verbose=True):
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

        traj_roll_mean = []
        traj_roll_std = []
        for obs in range(trajectory.shape[0]):
            step = 1
            time_in_w = timedelta(0)
            while (time_in_w.seconds <= epsilon) and obs+step < trajectory.shape[0]:
                time_in_w = time_in_w + delta_T.iloc[obs+step]
                step = step + 1
            end = obs + step-1
            if np.isnan(trajectory.iloc[obs:end].mean()):
                end = end + 1
            traj_roll_mean.append(trajectory.iloc[obs:end].mean())
            traj_roll_std.append(trajectory.iloc[obs:end].std())
            mean_window_traj.append(step)

        traj_roll = pd.DataFrame()
        traj_roll[f'ma_t_{col}'] = pd.DataFrame(traj_roll_mean)
        traj_roll[f'ms_t_{col}'] = pd.DataFrame(traj_roll_std)
        data_agg = pd.concat([data_agg, traj_roll], axis=0)
        mean_window.append(np.array(mean_window_traj).mean())

    if verbose:
        print(f'\t time window: {np.array(mean_window).mean()}, {np.array(mean_window).std()}')
    data_agg = data_agg.reset_index(drop=True)
    return data_agg


def get_features(data, n_dirs=4, win=30, eps=180):
    features = remove_short_trajectories(data)
    features = features[['trajectory', 'time', 'sog', 'cog']]
    features['rose'] = cog_windrose(features['cog'], n_dirs=n_dirs)
    # encoding
    enc_rose = pd.get_dummies(features['rose'])
    features = pd.concat([features, enc_rose], axis=1)

    features['roc'] = diff_cog(features['cog'])
    features['acceleration'] = features['sog'].diff()
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
