# coding=utf-8
#
#  Copyright 2022, Gabriel Spadon, all rights reserved.
#  This code is under GNU General Public License v3.0.
#      www.spadon.com.br & gabriel@spadon.com.br
#
# This script requires setting "CUBLAS_WORKSPACE_CONFIG=:16:8" as an environment variable.

import torch
import random
import numpy as np
import pandas as pd

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
	"window": 10,
	"variables": 4,
	"shuffle": True,
	"verbose": True,
	"batch_size": 256,
	"hidden_size": 128,
	"test_samples": 30,
	"use_amsgrad": True,
	"max_gradnorm": 1.0,
	"tuning_samples": 15,
	"weight_decay": 0.01,
	"recurrent_layers": 1,
	"bidirectional": True,
	"normalize_data": True,
	"learning_rate": 0.001,
	"scheduler_patience": 3,
	"scheduler_factor": 0.9,
	"learning_patience": 10,
	"recurrent_unit": "LSTM",  # "RNN", "GRU", or "LSTM"
	"random_seed": random_seed,
	"improvement_threshold": 0.1,
}

# Martha: I changed it to the path where is the final results of the clustering labelling
df_time = pd.read_csv("../results/time_final/fishing_8_600.csv")
df_obs = pd.read_csv("../results/observations_final/fishing_8_10.csv")


def batchfy_data(df, window=hyperparameters["window"]):
	x, y = [], []
	for mmsi in set(df.mmsi):
		df_mmsi = df[df.mmsi == mmsi]
		if df_mmsi.shape[0] >= window:
			df_mmsi = df_mmsi.sort_values("time")
			df_mmsi = df_mmsi[["lat", "lon", "sog", "cog", "labels"]]
			x.append(torch.from_numpy(df_mmsi.loc[:, df_mmsi.columns != "labels"].to_numpy()))
			y.append(torch.from_numpy(df_mmsi.labels.to_numpy()))
	return x, y


# print("\n>>> Features #1")
print("\n>>> Observation-based #1")
x, y = batchfy_data(df_obs)
mynn = NetworkPlayground(**hyperparameters).cuda()
mynn.fit(x, y)

# print("\n>>> Features #2")
print("\n>>> Time-based #2")
x, y = batchfy_data(df_time)
mynn = NetworkPlayground(**hyperparameters).cuda()
mynn.fit(x, y)

# print("\n>>> Features #3")
# x, y = batchfy_data(dfc)
# mynn = NetworkPlayground(**hyperparameters).cuda()
# mynn.fit(x, y)
