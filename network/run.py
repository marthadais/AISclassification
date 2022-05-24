# coding=utf-8
#
#  Copyright 2022, Gabriel Spadon, all rights reserved.
#  This code is under GNU General Public License v3.0.
#      www.spadon.com.br & gabriel@spadon.com.br
#
# This script requires setting "CUBLAS_WORKSPACE_CONFIG=:16:8" as an environment variable.
import multiprocessing

import os
import torch
import random
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

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
	"dropout": .0,
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
	"scheduler_patience": 2,
	"scheduler_factor": 0.8,
	"learning_patience": 6,
	"recurrent_unit": "LSTM",  # "RNN", "GRU", or "LSTM"
	"random_seed": random_seed,
	"improvement_threshold": 0.1,
}

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

def test_checkpoint(hyperparams, filename):
	df = pd.read_csv("../results/time_final/fishing_8_600.csv")
	# df = pd.read_csv("../results/observations_final/fishing_8_10.csv")
	hyperparameters.update(hyperparams)  # for the current iteration
	hyperparameters["details"] = deepcopy(hyperparams)  # for identification
	return (NetworkPlayground(**hyperparameters).cuda()).test_checkpoint(*batchfy_data(df), filename)

def test_pipelines(hyperparams):
	try:
		hyperparameters.update(hyperparams)  # for the current iteration
		hyperparameters["details"] = deepcopy(hyperparams)  # for identification

		# Dropout works only when two or more RNN layers are used, but never in the last layer
		if hyperparameters["suffix"] == "T":  # time-based pipeline
			df = pd.read_csv("../results/time_final/fishing_8_600.csv")
		elif hyperparameters["suffix"] == "O":  # observation-based pipeline
			df = pd.read_csv("../results/observations_final/fishing_8_10.csv")
		else:
			raise Exception("Suffix must be either 'T' or 'O'")

		torch.cuda.empty_cache()
		return (NetworkPlayground(**hyperparameters).cuda()).fit(*batchfy_data(df))

	except Exception as e:
		torch.cuda.empty_cache()
		print(e)

	torch.cuda.empty_cache()
	return None

search_space = {
	"verbose": [False],  # recommended during debugging
	"batch_size": [4096],  # varies with the GPU Memory
	"dropout": [.0, .15],  # RNN's dropout probability
	"suffix": ["T", "O"],  # different datasets (do not change)
	"window": [8, 9, 10],  # according to the unsupervised analysis
	"recurrent_layers": [1, 2, 3],  # number of stacked recurrent layers
	"bidirectional": [True, False],  # temporal-dependency direction
	"hidden_size": [64, 128, 256],  # size of the hidden layers
	"recurrent_unit": ["LSTM", "RNN", "GRU"],  # different RNNs
}  # This is a comprehensive, but reduced, set of possibilities

queries = []
details = [["suffix", "window", "dropout", "hidden_size", "recurrent_layers",
            "recurrent_unit", "bidirectional", "min_loss", "filename"]]
for f in os.listdir("./training-checkpoints/"):
	checkpoint = torch.load(os.path.join("./training-checkpoints/", f))
	query = checkpoint["details"]; queries.append(checkpoint["details"])
	details.append([query["suffix"], query["window"], query["dropout"],
	                query["hidden_size"], query["recurrent_layers"],
	                query["recurrent_unit"], query["bidirectional"],
	                checkpoint["min_loss"], f])

if len(details) > 1:
	# Save the compiled results of the models tested this far
	results_df = pd.DataFrame(details[1:], columns=details[0])
	shared_columns = list(set(results_df.columns) - {"min_loss", "filename"})
	compiled_df = results_df.loc[results_df.groupby(shared_columns).min_loss.idxmin()]
	compiled_df = compiled_df[((compiled_df["dropout"] == 0) | ((compiled_df["dropout"] > 0) & (compiled_df["recurrent_layers"] > 1)))]
	compiled_df.sort_values("min_loss").to_csv("compiled-results.csv")

	for duplicate in set(results_df.filename.values) - set(compiled_df.filename.values):
		os.remove(os.path.join("./training-checkpoints/", duplicate))

temp_search = [dict(zip(search_space, x)) for x in itertools.product(*search_space.values())]  # Complete in-grid search space
grid_search = [q for q in temp_search if (q["dropout"] == 0 or (q["dropout"] > 0 and q["recurrent_layers"] > 1)) and (q not in queries)]
np.random.shuffle(grid_search)  # Randomly shuffle for increased variability of the experiments during the early stages

# Benchmarking results regarding the temporal approach
print("The search space size is of %d possibilities!" % len(grid_search))
_ = [test_pipelines(params) for params in grid_search]
