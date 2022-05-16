import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_bars(stats_per_day, cols_pv, name_pv):
    fig = plt.figure(figsize=(10, 7))
    lbl = [f'{i}' for i in cols_pv]
    aux = stats_per_day[lbl]
    plt.bar(lbl, aux.sum(), color=col)
    plt.xlabel(f'{name_pv}', fontsize=14)
    plt.ylabel('Counting', fontsize=14)
    plt.grid(True)
    plt.show()
    plt.close()


def plot_lines(stats_per_day, cols_pv):
    fig = plt.figure(figsize=(25, 7))
    i = 0
    for iter in cols_pv:
        iter_aux = stats_per_day[iter]
        plt.plot(stats_per_day.index, iter_aux, color=col[i], marker="p", linestyle="-", linewidth=2, markersize=7, label=iter)
        i = i + 1
    plt.ylabel('Count', fontsize=20)
    plt.xlabel('Dates', fontsize=20)
    plt.legend(fontsize=20)
    plt.xticks(iter_aux.index, iter_aux.index, fontsize=15, rotation=90)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()
    # plt.savefig(f'{folder}/lines-compression-{item}.png', bbox_inches='tight')
    plt.close()


def calc_time(dayset):
    vessel_time = pd.DataFrame(dayset.to_list(), columns=['trajectory', 'time', 'status'])
    # compute the differences between the current and the previous information
    mean_t = 0
    for i in vessel_time['trajectory'].unique():
        aux = vessel_time[vessel_time['trajectory'] == i]
        if aux.shape[0] < 2:
            mean_t = mean_t + 0
        else:
            time_diff = aux['time'].diff().dt.seconds/3600
            mean_t = mean_t + time_diff.sum()
    return mean_t/len(vessel_time['trajectory'].unique())


def calc_count_fishing(dayset):
    vessel_time = pd.DataFrame(dayset.to_list(), columns=['trajectory', 'time', 'status'])
    # compute the differences between the current and the previous information
    count = 0
    for i in vessel_time['trajectory'].unique():
        aux = vessel_time[vessel_time['trajectory'] == i]
        if (aux['status'] == 7).sum() > 0:
            count = count + 1
    return count


def calc_count_non_fishing(dayset):
    vessel_time = pd.DataFrame(dayset.to_list(), columns=['trajectory', 'time', 'status'])
    # compute the differences between the current and the previous information
    count = 0
    for i in vessel_time['trajectory'].unique():
        aux = vessel_time[vessel_time['trajectory'] == i]
        if (aux['status'] == 7).sum() == 0:
            count = count + 1
    return count

# data_file = f'../data/preprocessed/DCAIS_[30_ 1001_ 1002]_None-mmsi_region_[46_ 51_ -130_ -122.5]_01-04_to_30-06_trips.csv'
# data_file = f'../data/preprocessed/DCAIS_[30, 50, 60, 37]_None-mmsi_region_[46, 51, -130, -122.5]_01-04_to_30-04_trips.csv'
data_file = f'../data/preprocessed/DCAIS_[30]_None-mmsi_region_[46, 51, -130, -122.5]_01-04_to_30-09_trips.csv'
# data_file = f'../fishing_10_10.csv'
dataset = pd.read_csv(data_file, parse_dates=['time'], low_memory=False)
dataset['time'] = dataset['time'].astype('datetime64[ns]')
dataset = dataset.sort_values(by=['trajectory', "time"])

print('Statistics')

dataset['day'] = dataset['time'].dt.date
# per status
dataset['combined_time'] = dataset[['trajectory', 'time', 'status']].values.tolist()
#getting time period
x = dataset['time'].dt.hour
b = [0,5,11,17,24]
period = ['Dawn', 'Morning', 'Afternoon', 'Night']
aux = pd.cut(x, bins=b, labels=period, include_lowest=True)
dataset['period'] = aux

vessel_type = dataset['vessel_type'].unique().tolist()

stats_per_day = pd.DataFrame()
stats_per_day['count'] = dataset[['day', 'trajectory']].groupby('day')['trajectory'].nunique()



for fl in dataset['flag'].unique():
    stats_per_day[f'{fl}'] = dataset[dataset['flag'] == fl].groupby('day')['trajectory'].nunique()
for vt in vessel_type:
    data_aux = dataset[dataset['vessel_type'] == vt]
    stats_per_day[f'{vt}'] = data_aux.groupby('day')['trajectory'].nunique()
    stats_per_day[f'{vt}-fishing'] = data_aux[data_aux['status'] == 7].groupby('day')['combined_time'].agg(calc_count_fishing)
    stats_per_day[f'{vt}-non-fishing'] = data_aux[data_aux['status'] != 7].groupby('day')['combined_time'].agg(calc_count_non_fishing)

for pr in period:
    data_aux = dataset[dataset['period'] == pr]
    stats_per_day[f'{pr}'] = data_aux.groupby('day')['trajectory'].nunique()
    stats_per_day[f'{pr}-fishing'] = data_aux[data_aux['status'] == 7].groupby('day')['combined_time'].agg(calc_count_fishing)
    stats_per_day[f'{pr}-non-fishing'] = data_aux[data_aux['status'] != 7].groupby('day')['combined_time'].agg(calc_count_non_fishing)

stats_per_day[f'fishing'] = dataset[dataset['status'] == 7].groupby('day')['combined_time'].agg(calc_time)
stats_per_day[f'non-fishing'] = dataset[dataset['status'] != 7].groupby('day')['combined_time'].agg(calc_time)

stats_per_day = stats_per_day.fillna(0)

col = ['black', 'blue', 'green', 'darkorange', 'crimson', 'purple', 'orange', 'red']

plot_bars(stats_per_day, period, 'Period')
# plot_bars(stats_per_day, vessel_type, 'Vessel type')

fig = plt.figure(figsize=(10, 7))
lbl1 = [f'{i}-fishing' for i in period]
lbl2 = [f'{i}-non-fishing' for i in period]
aux1 = stats_per_day[lbl1]
aux2 = stats_per_day[lbl2]
x_axis = np.arange(len(aux1.sum()))
plt.bar(x_axis-0.2, aux1.sum(), 0.4, label='fishing')
plt.bar(x_axis+0.2, aux2.sum(), 0.4, label='non-fishing')
plt.xticks(x_axis, period)
plt.xlabel('Period', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.legend()
plt.show()
plt.close()

# per flag
plot_lines(stats_per_day, dataset['flag'].unique())
# plot_lines(dataset, 'vessel_type')
plot_lines(stats_per_day, period)

cols_name = [f'{pr}-fishing' for pr in period]
plot_lines(stats_per_day, cols_name)


fig = plt.figure(figsize=(25, 7))
fl_aux = dataset[dataset['status'] == 7].groupby('day')['combined_time'].agg(calc_time)
plt.plot(fl_aux.index, fl_aux, color=col[0], marker="p", linestyle="-", linewidth=2,
             markersize=7, label='fishing')
fl_aux = dataset[dataset['status'] != 7].groupby('day')['combined_time'].agg(calc_time)
plt.plot(fl_aux.index, fl_aux, color=col[1], marker="p", linestyle="-", linewidth=2,
             markersize=7, label='non-fishing')
plt.ylabel('Time spent (hours)', fontsize=20)
plt.xlabel('Dates', fontsize=20)
plt.legend(fontsize=20)
plt.xticks(fl_aux.index, fl_aux.index, fontsize=15, rotation=90)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.show()
# plt.savefig(f'{folder}/lines-compression-{item}.png', bbox_inches='tight')
plt.close()


fig = plt.figure(figsize=(25, 7))
i=0
for p in period:
    data_aux = dataset[dataset['period'] == p]
    fl_aux = data_aux[data_aux['status'] == 7].groupby('day')['combined_time'].agg(calc_time)
    plt.plot(fl_aux.index, fl_aux, color=col[i], marker="p", linestyle="-", linewidth=2,
             markersize=7, label='fishing')
    i = i+1
    fl_aux = data_aux[data_aux['status'] != 7].groupby('day')['combined_time'].agg(calc_time)
    plt.plot(fl_aux.index, fl_aux, color=col[i], marker="p", linestyle="-", linewidth=2,
             markersize=7, label='non-fishing')
    i = i+1
plt.ylabel('Time spent (hours)', fontsize=20)
plt.xlabel('Dates', fontsize=20)
plt.legend(fontsize=20)
plt.xticks(fl_aux.index, fl_aux.index, fontsize=15, rotation=90)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.show()
# plt.savefig(f'{folder}/lines-compression-{item}.png', bbox_inches='tight')
plt.close()

def images_eval():
    import matplotlib.pyplot as plt
    from ast import literal_eval
    import numpy as np

    eval_obs = pd.read_csv(f'./results/observations/measures.csv', index_col=0)
    eval_time = pd.read_csv(f'./results/time/measures.csv', index_col=0)

    for iter in range(2, 21):
        # iter = 8
        seconds = iter*60
        obs_d = eval_obs.loc[:, str(iter)].apply(lambda x: literal_eval(x)[0])
        time_d = eval_time.loc[:, str(seconds)].apply(lambda x: literal_eval(x)[0])
        index_d = eval_obs.index
        # obs_d = eval_obs.loc[iter, :].apply(lambda x: literal_eval(x)[0])
        # time_d = eval_time.loc[iter, :].apply(lambda x: literal_eval(x)[0])
        # index_d = eval_obs.columns

        fig = plt.figure(figsize=(10, 7))
        plt.plot(index_d, time_d, marker="p", linestyle="-", linewidth=2,
                 markersize=7, label='time-based')
        plt.plot(index_d, obs_d, marker="p", linestyle="-", linewidth=2,
                 markersize=7, label='message-based')
        plt.plot(index_d, np.repeat(0.5, len(obs_d)), color='black', linestyle="-", linewidth=2, label='threshold')
        plt.title(f'K-means - {iter} obs')
        plt.ylabel('DBI', fontsize=15)
        # plt.xlabel('Number of messages and minutes', fontsize=15)
        plt.xlabel('Number of clusters', fontsize=15)
        plt.legend(fontsize=20)
        plt.xticks(index_d, index_d, fontsize=15)
        plt.yticks(np.arange(0.38, 0.55, step=0.02), fontsize=15)
        plt.tight_layout()
        plt.show()
        # plt.savefig(f'../results/DBI_{iter}.png', bbox_inches='tight')
        # plt.close()

    # average n cluster
    line_obs = []
    line_time = []
    for iter in range(5, 16):
        seconds = iter*60
        obs_d = eval_obs.loc[:, str(iter)].apply(lambda x: literal_eval(x)[0])
        time_d = eval_time.loc[:, str(seconds)].apply(lambda x: literal_eval(x)[0])
        index_d = eval_obs.columns
        line_obs.append(obs_d.mean())
        line_time.append(time_d.mean())

    fig = plt.figure(figsize=(10, 7))
    plt.plot(index_d, line_time, marker="p", linestyle="-", linewidth=2,
             markersize=7, label='time')
    plt.plot(index_d, line_obs, marker="p", linestyle="-", linewidth=2,
             markersize=7, label=f'observation')
    # plt.title(f'K-means - {iter} clusters')
    plt.ylabel('Average of DBI', fontsize=15)
    plt.xlabel('Number of observation and minutes', fontsize=15)
    plt.legend(fontsize=20)
    plt.xticks(index_d, index_d, fontsize=15)
    plt.yticks(np.arange(0.47, 0.51, step=0.005), fontsize=15)
    plt.tight_layout()
    plt.show()


    # average n obs
    line_obs = []
    line_time = []
    for iter in range(2, 16):
        obs_d = eval_obs.loc[iter, :].apply(lambda x: literal_eval(x)[0])
        time_d = eval_time.loc[iter, :].apply(lambda x: literal_eval(x)[0])
        index_d = eval_obs.index
        line_obs.append(obs_d.mean())
        line_time.append(time_d.mean())

    fig = plt.figure(figsize=(10, 7))
    plt.plot(index_d, line_time, marker="p", linestyle="-", linewidth=2,
             markersize=7, label='time')
    plt.plot(index_d, line_obs, marker="p", linestyle="-", linewidth=2,
             markersize=7, label=f'observation')
    # plt.title(f'K-means - {iter} clusters')
    plt.ylabel('Average of DBI', fontsize=15)
    plt.xlabel('Number of clusters', fontsize=15)
    plt.legend(fontsize=20)
    plt.xticks(index_d, index_d, fontsize=15)
    plt.yticks(np.arange(0.40, 0.55, step=0.01), fontsize=15)
    plt.tight_layout()
    plt.show()


    # plt.savefig(f'{folder}/lines-compression-{item}.png', bbox_inches='tight')
    # plt.close()

    # fig = plt.figure(figsize=(10, 7))
    # plt.plot(eval_obs.columns, eval_time.loc[3, :], marker="p", linestyle="-", linewidth=2,
    #          markersize=7, label='time')
    # plt.plot(eval_obs.columns, eval_obs.loc[3, :], marker="p", linestyle="-", linewidth=2,
    #          markersize=7, label='obs')
    # plt.title('Pos-processing')
    # plt.ylabel('DBI', fontsize=20)
    # plt.xlabel('Number of clusters', fontsize=20)
    # plt.legend(fontsize=20)
    # plt.xticks(eval_obs.columns, eval_obs.columns, fontsize=15, rotation=90)
    # plt.yticks(fontsize=20)
    # plt.tight_layout()
    # plt.show()
    # plt.savefig(f'{folder}/lines-compression-{item}.png', bbox_inches='tight')
    # plt.close()
