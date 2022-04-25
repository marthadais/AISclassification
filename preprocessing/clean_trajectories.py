import os, random
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from datetime import timedelta
import pickle
import warnings
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def missing_values_treatment(x):
    """
    It removes observations with invalid SOG and COG.
    :param x: the dataset
    :return: the dataset without invalid samples
    """
    # missing and invalid values in sog and cog are removed
    if 'sog' in x.columns:
        x.loc[x['sog'] == 'None', 'sog'] = -1
        x.sog = x.sog.astype('float64')
        x = x.drop(x[x['sog'] == -1].index)
        # remove messages when it is not moving
        x.loc[x['sog'] <= 0.5, 'sog'] = -1
        x.sog = x.sog.astype('float64')
        x = x.drop(x[x['sog'] == -1].index)

    if 'cog' in x.columns:
        x.loc[x['cog'] < 0, 'cog'] = -1
        x.loc[x['sog'] == 0, 'cog'] = -1
        x.loc[x['cog'] == 'None', 'cog'] = -1
        x.cog = x.cog.astype('float64')
        x = x.drop(x[x['cog'] == -1].index)

    return x


def removing_invalid_samples(x, min_obs=None, subset=None):
    """
    It round the values to 4 decimals, removes duplicates, and removes samples with few observations.
    :param x: the dataset
    :return: the dataset with country attribute
    """
    # round values to 4 decimals (10 meters)
    x.lat = x.lat.round(4)
    x.lon = x.lon.round(4)

    # remove duplicate entries
    x = x.drop_duplicates(subset=subset, keep='first')

    x = missing_values_treatment(x)

    if min_obs is not None:
        # remove mmsi with less than min observations
        obs_per_mmsi = x.groupby(x['mmsi'], as_index=False).size()
        ids_to_keep = obs_per_mmsi['mmsi'][obs_per_mmsi['size'] >= min_obs]
        x = x[x['mmsi'].isin(ids_to_keep)]

    return x


def include_country(x):
    """
    It includes the country based on the MMSI
    :param x: the dataset
    :return: the dataset with country attribute
    """
    # include flags
    MMIS_digits_path = './data/MaritimeIdentificationDigits.csv'
    if os.path.exists(MMIS_digits_path):
        MID = pd.read_csv(MMIS_digits_path)
        flag_col = x['mmsi']
        flag_col = flag_col // 1000000
        flag_col = flag_col.replace(MID.set_index('Digit')['Allocated to'])
        x = x.assign(flag=pd.Series(flag_col, index=x.index))
    else:
        warnings.warn(f'File {MMIS_digits_path} was not found.')
    return x


### Reading and filtering dataset ###
def date_range(start_date, end_date):
    """
    It provides ranges of date period to conduct the loop
    :param start_date: initial date period to get the dataset
    :param end_date: final date period to get the dataset
    :return: iterative data
    """
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def create_dataset_noaa(path, time_period, vt=None):
    """
    It reads the noaa dataset and produce a csv file with the vessels information of a specific type.
    Such vessel type provide the most trips information.
    :param time_period: initial and final date period to get the dataset
    :param vt: vessel type
    :return: path where the csv file was saved
    """
    columns_read = ['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG', 'Status', 'VesselType']
    dataset = pd.DataFrame()
    mmsis = np.array([])
    for curr_date in date_range(time_period[0], time_period[1]+timedelta(days=1)):
        print(f'\treading day {curr_date}')
        url = urlopen(f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2020/AIS_2020_{curr_date.month:02d}_{curr_date.day:02d}.zip")
        file_name = f'AIS_2020_{curr_date.month:02d}_{curr_date.day:02d}.csv'
        zipfile = ZipFile(BytesIO(url.read()))
        chunk = pd.read_csv(zipfile.open(file_name), usecols=columns_read)
        if vt is not None:
            # chunk2 = chunk[chunk['VesselType'] == vt]
            chunk2 = chunk[chunk['VesselType'].isin(vt)]
            mmsis = np.concatenate((mmsis, chunk2['MMSI'].unique()))
            mmsis = mmsis.astype(float)
            mmsis = np.unique(mmsis)
            chunk = chunk[chunk['MMSI'].isin(mmsis)]
        dataset = pd.concat([dataset, chunk], ignore_index=True)
        zipfile.close()

    dataset['VesselType'] = dataset['VesselType'].fillna(-1)
    dataset['VesselType'] = dataset['VesselType'].astype(int)

    dataset = dataset[['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG', 'Status', 'VesselType']]
    dataset.columns = ['mmsi', 'time', 'lat', 'lon', 'sog', 'cog', 'status', 'vessel_type']

    dataset.to_csv(path, index=False)


def dict_to_pandas(dataset):
    """
    It converts the dict dataset into pandas format.
    :return: dataset in a pandas format.
    """
    new_dataset = pd.DataFrame()
    ids = dataset.keys()
    for i in ids:
        curr_traj = pd.DataFrame.from_dict(dataset[i])
        new_dataset = pd.concat([new_dataset, curr_traj], axis=0)
    return new_dataset

### Class to produce the preprocessed dataset ###
class Trajectories:
    """
    It reads the DCAIS dataset and produce a csv file with the preprocessed vessels information.
    The dataset corresponds to a specific vessel type for a particular period of time.
    It reads, clean and aggregate information of the vessels.
    """
    def __init__(self, n_samples=None, vessel_type=None, time_period=None, min_obs=100, **args):
        """
        It reads the noaa dataset and produce a csv file with the vessels information of a specific type.
        Such vessel type provide the most trips information.
        :param n_samples: number of MMSI to be processed, none if you request all MMSIs (Default: None)
        :param vessel_typet: vessel type
        :param time_period: period of time to read the dataset
        :param min_obs: minimum number of observations
        """
        self._nsamples = n_samples
        self._vt = vessel_type
        self._vessel_types = None
        self._columns_set = ['lat', 'lon', 'cog', 'sog', 'status']
        # it just considers trajectories with more than such number of observations
        self.min_obs = min_obs
        if self.min_obs < 2:
            self.min_obs = 2

        self.region = None
        if 'region' in args.keys():
            self.region = args['region']

        self.compress = None
        if 'compress' in args.keys():
            self.compress = args['compress']

        if time_period is None:
            time_period = (datetime(2020, 4, 19), datetime(2020, 4, 25))

        if not os.path.exists('./data/preprocessed/'):
            os.makedirs('./data/preprocessed/')

        self._day_name = f'{time_period[0].day:02d}-{time_period[0].month:02d}_to_{time_period[1].day:02d}-{time_period[1].month:02d}'
        self.dataset_path = f"./data/preprocessed/DCAIS_{self._vt}_{self._day_name}_time_period.csv"
        self.cleaned_path = f"./data/preprocessed/DCAIS_{self._vt}_{self._day_name}_clean.csv"
        self.preprocessed_path = f"./data/preprocessed/DCAIS_{self._vt}_{self._nsamples}-mmsi_{self._day_name}_trips.csv"
        if self.region is not None:
            self.preprocessed_path = f"./data/preprocessed/DCAIS_{self._vt}_{self._nsamples}-mmsi_region_{self.region}_{self._day_name}_trips.csv"

        self.compress_path = None
        self.compress_rate_path = None
        self.time_rate_path = None

        if not os.path.exists(self.dataset_path):
            create_dataset_noaa(self.dataset_path, vt=self._vt, time_period=time_period)
            print(f'Preprocessed data save at: {self.dataset_path}')

        if not os.path.exists(self.cleaned_path):
            self._cleaning()
            print(f'Clean data save at: {self.cleaned_path}')

        if not os.path.exists(self.preprocessed_path):
            self._mmsi_trips()
            print(f'Preprocessed trips data save at: {self.preprocessed_path}')

    def _cleaning(self):
        """
        It cleans the dataset, removing invalid samples and including country information.
        """
        # reading dataset of a time period
        dataset = pd.read_csv(self.dataset_path, parse_dates=['time'])
        dataset['time'] = dataset['time'].astype('datetime64[ns]')
        dataset = dataset.sort_values(['time'])

        # removing invalid data
        dataset = removing_invalid_samples(dataset, min_obs=self.min_obs, subset=['mmsi', 'time'])
        # including country information
        dataset = include_country(dataset)
        dataset.to_csv(self.cleaned_path, index=False)

    def _mmsi_trips(self):
        """
        It reads the DCAIS dataset, select MMSI randomly if a number of samples is defined.
        It process the trajectories of each MMSI (pandas format).
        Save the dataset in a csv file.
        """
        # reading dataset of a time period
        dataset = pd.read_csv(self.cleaned_path, parse_dates=['time'])
        dataset['time'] = dataset['time'].astype('datetime64[ns]')
        dataset = dataset.sort_values(by=['mmsi', "time"])

        # select mmsi randomly
        ids = dataset['mmsi'].unique()
        new_dataset = pd.DataFrame()
        # create trajectories
        count_mmsi = 0
        count_traj = 0
        for id in ids:
            print(f'\t Preprocessing each trajectory {count_mmsi} of {len(ids)}')
            trajectory = dataset[dataset['mmsi'] == id]

            # selecting the region
            isin_region = True
            if self.region is not None:
                if ((trajectory['lon'] >= self.region[2]) & (trajectory['lon'] <= self.region[3]) & (trajectory['lat'] >= self.region[0]) & (trajectory['lat'] <= self.region[1])).sum() == 0:
                    isin_region = False

            # if is inside the selected region and contains enough observations
            if isin_region:
                crop=True
                # observations that is in the region
                if crop==True:
                    trajectory_crop = trajectory[
                        (trajectory['lon'] >= self.region[2]) & (trajectory['lon'] <= self.region[3]) & (
                                trajectory['lat'] >= self.region[0]) & (trajectory['lat'] <= self.region[1])]
                else:
                    trajectory_crop = trajectory

                # include trajectory id
                aux_col = pd.DataFrame({'trajectory': np.repeat(count_traj, trajectory_crop.shape[0])})
                trajectory_crop.reset_index(drop=True, inplace=True)
                trajectory_crop = pd.concat([aux_col, trajectory_crop], axis=1)
                # add trajectory
                new_dataset = pd.concat([new_dataset, trajectory_crop], axis=0, ignore_index=True)
                count_traj = count_traj + 1
            count_mmsi = count_mmsi + 1

        if self._nsamples is not None:
            ids = new_dataset['mmsi'].unique()
            random.shuffle(ids)
            ids = ids[0:self._nsamples]
            new_dataset = new_dataset[new_dataset['mmsi'].isin(ids)]
        else:
            self._nsamples = count_traj

        if self.min_obs is not None:
            # remove mmsi with less than min observations
            obs_per_mmsi = new_dataset.groupby(new_dataset['mmsi'], as_index=False).size()
            ids_to_keep = obs_per_mmsi['mmsi'][obs_per_mmsi['size'] >= self.min_obs]
            new_dataset = new_dataset[new_dataset['mmsi'].isin(ids_to_keep)]

        new_dataset.to_csv(self.preprocessed_path, index=False)

    def get_dataset(self):
        """
        It converts the csv dataset into dict format.
        :return: dataset in a dict format.
        """
        dataset = pd.read_csv(self.preprocessed_path, parse_dates=['time'], low_memory=False)
        dataset['time'] = dataset['time'].astype('datetime64[ns]')
        dataset = dataset.sort_values(by=['trajectory', "time"])

        return dataset
