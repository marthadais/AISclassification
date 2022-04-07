# coding=utf-8
#
#  Copyright 2022, Gabriel Spadon, all rights reserved.
#  This code is under GNU General Public License v3.0.
#      www.spadon.com.br & gabriel@spadon.com.br
#
# For reproducibility set "CUBLAS_WORKSPACE_CONFIG=:16:8" as environment variable.

import torch
import random
import numpy as np

from network.architecture import NetworkPlayground


def random_data(n_samples, n_timestamps, n_features):
	"""
		Random data generator for unit testing.
		- arguments are self-explanatory
	"""
	shape = (n_samples, n_timestamps, n_features)
	x = (torch.expm1(torch.rand(shape) + 2) ** 2)
	y = torch.randint(low=0, high=2, size=shape[:-1])
	return x, y.to(torch.float32)


random_seed = 6723  # Same used inside the NN
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
torch.cuda.manual_seed_all(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

hyperparameters = {
	"bias": True,
	"window": 30,
	"variables": 5,
	"shuffle": True,
	"verbose": True,
	"batch_size": 256,
	"hidden_size": 64,
	"test_samples": 5,
	"use_amsgrad": True,
	"max_gradnorm": 1.0,
	"tuning_samples": 5,
	"weight_decay": 0.01,
	"recurrent_layers": 1,
	"bidirectional": False,
	"normalize_data": True,
	"learning_rate": 0.001,
	"scheduler_patience": 5,
	"scheduler_factor": 0.9,
	"learning_patience": 15,
	"recurrent_unit": "LSTM",  # "RNN", "GRU", or "LSTM"
	"random_seed": random_seed,
	"improvement_threshold": 0.1,
}

x, y = random_data(500, 1000, 5)
mynn = NetworkPlayground(**hyperparameters).cuda()
mynn.fit(x, y)
