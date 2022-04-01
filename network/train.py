import torch
import numpy as np
import pandas as pd

from network.architecture import MyNetwork


def random_data(n_samples, n_timestamps, n_features):
	"""
		Random data generator for unit testing.
		- arguments are self-explanatory
	"""
	shape = (n_samples, n_timestamps, n_features)
	x = (torch.expm1(torch.rand(shape) + 2) ** 2)
	y = torch.randint(low=0, high=2, size=shape[:-1])
	return x, y


# Data Preparation Test
# x, y = random_data(500, 1000, 5)
# mynn = MyNetwork(30, 5, 5, "LSTM", 64, 1, False, True)
# mynn.data_preparation(x, y)
