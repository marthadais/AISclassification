# coding=utf-8
#
#  Copyright 2022, Gabriel Spadon, all rights reserved.
#  This code is under GNU General Public License v3.0.
#      www.spadon.com.br & gabriel@spadon.com.br
#
# This script requires setting "CUBLAS_WORKSPACE_CONFIG=:16:8" as an environment variable.
import multiprocessing

import torch
import random
import itertools
import numpy as np
import pandas as pd

from tqdm.contrib.concurrent import process_map
from architecture import NetworkPlayground
from copy import deepcopy

random_seed = 6723  # Same used inside the NN
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
torch.cuda.manual_seed_all(random_seed)
torch.set_printoptions(sci_mode=False)
torch.set_default_dtype(torch.float64)
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

hyperparameters = {
	"bias": True,
	"window": 10,  # number of consecutive AIS messages to feed the NN (per mini-batch)
	"variables": 4,  # number of features per AIS message
	"shuffle": True,
	"verbose": True,
	"batch_size": 256,
	"hidden_size": 128,
	"test_samples": 50,  # refers to the number of unique trajectories reserved for test
	"use_amsgrad": True,
	"max_gradnorm": 1.0,
	"tuning_samples": 25,  # refers to the number of unique trajectories reserved for tuning
	"weight_decay": 0.01,
	"recurrent_layers": 1,
	"bidirectional": False,  # whether to assume a multidirectional temporal dependency in the trajectories
	"normalize_data": True,
	"learning_rate": 0.001,
	"scheduler_patience": 3,
	"scheduler_factor": 0.9,
	"learning_patience": 5,
	"recurrent_unit": "LSTM",  # "RNN", "GRU", or "LSTM"
	"random_seed": random_seed,
	"improvement_threshold": 0.1,
}


def test_pipelines(hyperparams):
	hyperparameters.update(hyperparams)  # for the current iteration
	hyperparameters["details"] = deepcopy(hyperparameters)  # for identification

	if hyperparameters["suffix"] == "T":  # time-based pipeline
		df = pd.read_csv("../results/time_final/fishing_8_600.csv")
	elif hyperparameters["suffix"] == "O":  # observation-based pipeline
		df = pd.read_csv("../results/observations_final/fishing_8_10.csv")
	else:
		raise Exception("Suffix must be either 'T' or 'O'")

	def batchfy_data(df, window=hyperparameters["window"]):
		"""
			Prepares the dataset for the neural network training.
			The testing portion will be separated by the trainer's class.
			To change the amount of test data, update "test_samples" in the dict above.
		"""
		x, y = [], []
		for mmsi in set(df.mmsi):
			df_mmsi = df[df.mmsi == mmsi]
			if df_mmsi.shape[0] >= window:
				df_mmsi = df_mmsi.sort_values("time")
				df_mmsi = df_mmsi[["lat", "lon", "sog", "cog", "labels_pos"]]
				x.append(torch.from_numpy(df_mmsi.loc[:, df_mmsi.columns != "labels_pos"].to_numpy()))
				y.append(torch.from_numpy(df_mmsi.labels_pos.to_numpy()))
		return x, y

	return (NetworkPlayground(**hyperparameters).cuda()).fit(*batchfy_data(df))


search_space = {  # essentially covering good possibilities
	"verbose": [False],  # recommended during debugging
	"batch_size": [4096],  # varies with the GPU Memory
	"suffix": ["T", "O"],  # different datasets (do not change)
	"window": [8, 9, 10, 11],  # according to the unsupervised analysis
	"recurrent_layers": [1, 2, 3],  # number of stacked recurrent layers
	"bidirectional": [True, False],  # temporal-dependency direction
	"hidden_size": [32, 64, 128, 256],  # size of the hidden layers
	"recurrent_unit": ["LSTM", "RNN", "GRU"],  # different RNNs
}  # This is a comprehensive, but reduced, set of possibilities
grid_search = [dict(zip(search_space, x)) for x in itertools.product(*search_space.values())]
np.random.shuffle(grid_search)  # Randomly shuffle for increased variability

# Benchmarking results regarding the temporal approach
# _ = process_map(test_pipelines, grid_search, max_workers=2, chunksize=250, disable=False)
print("The search space size is of %d possibilities!" % len(grid_search))
_ = [test_pipelines(params) for params in grid_search]
