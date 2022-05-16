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
import numpy as np
import pandas as pd

from tqdm.contrib.concurrent import thread_map
from network.architecture import NetworkPlayground

random_seed = 6723  # Same used inside the NN
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
torch.cuda.manual_seed_all(random_seed)
torch.set_printoptions(sci_mode=False)
torch.set_default_dtype(torch.float32)
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

hyperparameters = {
	"bias": True,
	"window": 10,  # number of consecutive AIS messages to feed the NN (per mini-batch) TODO: Should be defined in a grid search
	"variables": 4,  # number of features per AIS message
	"shuffle": True,
	"verbose": True,
	"batch_size": 256,
	"hidden_size": 128,  # TODO: Should be defined in a grid search
	"test_samples": 30,  # refers to the number of unique trajectories reserved for test
	"use_amsgrad": True,
	"max_gradnorm": 1.0,
	"tuning_samples": 15,  # refers to the number of unique trajectories reserved for tuning
	"weight_decay": 0.01,
	"recurrent_layers": 1,  # TODO: Should be defined in a grid search
	"bidirectional": True,  # whether to assume a multidirectional temporal dependency in the trajectories
	"normalize_data": True,
	"learning_rate": 0.001,
	"scheduler_patience": 3,
	"scheduler_factor": 0.9,
	"learning_patience": 10,
	"recurrent_unit": "LSTM",  # "RNN", "GRU", or "LSTM" TODO: Should be defined in a grid search
	"random_seed": random_seed,
	"improvement_threshold": 0.1,
}

def test_pipelines(func_input):
	df, suffix = func_input  # this is a list from the thread manager
	hyperparameters["suffix"] = suffix  # updating indexing information

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
				df_mmsi = df_mmsi[["lat", "lon", "sog", "cog", "labels"]]
				x.append(torch.from_numpy(df_mmsi.loc[:, df_mmsi.columns != "labels"].to_numpy()))
				y.append(torch.from_numpy(df_mmsi.labels.to_numpy()))
		return x, y

	# training the network with the input dataset
	(NetworkPlayground(**hyperparameters).cuda()).fit(**batchfy_data(df))

thread_map(test_pipelines, iter([(
	pd.read_csv("../results/observations_final/fishing_8_10.csv"),
	"OBS"  # Suffix for indexing the output
), (  # >>> add as many as necessary
	pd.read_csv("../results/time_final/fishing_8_600.csv"),
	"TIME"  # Suffix for indexing the output
)]), max_works=multiprocessing.cpu_count())
