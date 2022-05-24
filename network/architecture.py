# coding=utf-8
#
#  Copyright 2022, Gabriel Spadon, all rights reserved.
#  This code is under GNU General Public License v3.0.
#      www.spadon.com.br & gabriel@spadon.com.br
#
# This script requires setting "CUBLAS_WORKSPACE_CONFIG=:16:8" as an environment variable.

import os
import torch
import shutil
import random
import pprint
import torcheck
import datetime
import numpy as np

from tqdm import tqdm
from pprint import pprint
from datetime import datetime
from torch.optim import AdamW
from .center_loss import CenterLoss
from torch.nn import RNN, GRU, LSTM  # used with eval(<class-name>)
from prettytable import PrettyTable
from sklearn.metrics import f1_score
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau


class NetworkPlayground(torch.nn.Module):
	"""
	A Neural Network Playground for baselines computation.
	"""

	def __init__(self, window, variables, tuning_samples, test_samples, recurrent_unit, hidden_size, recurrent_layers, bidirectional, dropout, bias, **kwargs):
		"""
		Initializes the layers of the network with the user-input arguments.
		:param window: integer
			The number of time steps to look at in the past.
		:param window: integer
			The number of variables in the dataset.
		:param tuning_samples: integer
			The number of time-steps reserved for hyperparameter tuning.
		:param test_samples: integer
			The number of time steps to predict in the future.
		:param recurrent_unit: string
			Defines the architecture used as a recurrent layer between RNN, GRU, and LSTM.
		:param hidden_size: integer
			The number of elements in the hidden unit of the RNN.
		:param recurrent_layers: string
			The number of stacked recurrent unities.
		:param bidirectional: boolean
			Sets the recurrent unit as bidirectional if set to True.
		:param dropout: float
			Dropout probability.
		:param bias: boolean
			Sets bias vectors permanently to zeros if set to False.
		"""
		super(NetworkPlayground, self).__init__()
		torch.set_default_dtype(torch.float64)
		torch.set_printoptions(sci_mode=False)

		# Attributes
		self.epoch = 0
		self.bias = bias
		self.window = window
		self.min_loss = np.inf
		self.dropout = dropout
		self.variables = variables
		self.hidden_size = hidden_size
		self.test_samples = test_samples
		self.bidirectional = bidirectional
		self.tuning_samples = tuning_samples
		self.recurrent_unit = recurrent_unit
		self.recurrent_layers = recurrent_layers

		# Evaluation Metrics
		self.BCE = torch.nn.BCELoss()

		for key, value in kwargs.items():
			# Mapping all kwargs to attributes
			setattr(self, key, value)

		# Assuring the unicity of the experiments
		self.hash = round(datetime.now().timestamp())
		self.hash += random.randint(0, 999)

		# Neural Network Layers
		self.RNN = eval(recurrent_unit)(
				bidirectional=bidirectional, bias=bias,
				input_size=variables, hidden_size=hidden_size,
				num_layers=recurrent_layers, dropout=dropout, batch_first=True)
		self.Linear1 = torch.nn.Linear(in_features=window, out_features=hidden_size, bias=bias)
		self.Linear2 = torch.nn.Linear(in_features=hidden_size, out_features=1, bias=False)
		self.Linear3 = torch.nn.Linear(in_features=hidden_size, out_features=1, bias=False)
		self.Sigmoid = torch.nn.Sigmoid()
		self.ReLU = torch.nn.ReLU()

		# Training Defines
		self.criterion = self.BCE
		self.center_loss = CenterLoss(classes=2, variables=variables).cuda()
		shared_parameters = list(self.parameters()) + list(self.criterion.parameters())
		self.optimizer = AdamW(shared_parameters, lr=self.learning_rate, weight_decay=self.weight_decay, amsgrad=self.use_amsgrad)
		self.scheduler = ReduceLROnPlateau(self.optimizer, factor=self.scheduler_factor, patience=self.scheduler_patience, threshold=self.improvement_threshold)

		# Further Debugging with Torcheck
		torcheck.register(self.optimizer)
		torcheck.add_module(
				module=self,
				changing=True,
				check_nan=True,
				check_inf=True,
				module_name="NetworkPlayground",
		)

	def forward(self, x):
		"""
		Defines the computation performed at every call.
		:param x: array-like of shape (samples, window, variables)
			Observations from the past window-sized time-steps.
		:return: array-like of shape (samples, horizon, variables)
			Predictions for the next horizon-sized time-steps.
		"""
		x, _ = self.RNN(x)
		if self.bidirectional:
			x = x.view(-1, self.window, 2, self.hidden_size)
			x = x.sum(axis=2)  # merging temporal branches into one branch
		x = self.Linear1(x.permute(0, 2, 1))  # encoding the temporal axis
		x = self.Linear2(x.permute(0, 2, 1))  # decoding the variable axis
		x = self.ReLU(x)  # introducing non-linearity in the decoding process
		x = self.Linear3(x.permute(0, 2, 1))  # decoding the temporal axis
		return self.Sigmoid(torch.squeeze(x))  # compressing dimensions

	def __fit(self, x, y, alpha=1.75, center_lr=.25):
		"""
		PyTorch training routine.
		:param x: array-like of shape (samples, window, variables)
			Observations from the past window-sized time-steps.
		:param y: array-like of shape (samples, horizon, variables)
			Predictions for the next horizon-sized time-steps.
		:param alpha: integer
			Weighting parameter for the center loss.
		:param center_lr: float
			Learning rate for the center loss.
		:return: array-like of shape (1,)
			The criterion loss.
		"""
		self.train()
		self.optimizer.zero_grad()

		# Forward propagation
		y_pred = self.forward(x)
		# Computing the resulting loss
		entropy_loss = self.criterion(y_pred, y)
		center_loss = (self.center_loss(x, y) * alpha)
		(entropy_loss + center_loss).backward()  # joint backprop

		for param in self.center_loss.parameters():
			# Weighting the loss function parameters based on alpha
			param.grad.data *= (center_lr / (alpha * self.learning_rate))

		# Gradient value clipping
		torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_gradnorm)
		self.optimizer.step()  # updating parameters for next call
		return entropy_loss.detach().cpu().numpy()

	def predict(self, x):
		"""
		PyTorch inference routine.
		:param x: array-like of shape (samples, window, variables)
			Observations from the past window-sized time-steps.
		:return: array-like of shape (samples,)
			One label for each sample.
		"""
		self.eval()
		with torch.no_grad():
			return torch.round(self.forward(x).cpu())

	def data_preparation(self, xin_list, yin_list, generator, num_workers=cpu_count(), drop_last=True):
		"""
		Iterates over the time axis, segmenting the time series for neural network training.
		:param xin_list: a list of tensors of shape (n_timestamps, n_features)
			A tensor of float-valued data features to use for classification.
		:param yin_list: a list of tensors of shape (n_timestamps, n_features)
			A tensor of integer-valued monotonically-increasing labels.
		:param generator: object
			Generator that manages the state of the pseudo-random numbers algorithm.
		:param num_workers: integer
			How many subprocesses to use for data loading.
		:param drop_last: boolean
			Set to True to drop the last batch if incomplete.
		"""
		xf, yf, xt, yt, xd, yd = [], [], [], [], [], []

		for x, y in zip(xin_list, yin_list):
			slice_ids = [[], [], []]  # used for validation
			try:  # will fail whenever the series is too small on the temporal axis

				xt_list, yt_list = [], []
				# Isolating the test data from the end of the dataset
				w_start, w_end = (x.shape[0] - self.window, x.shape[0])
				for _ in range(self.test_samples):
					assert w_start >= 0, "Invalid arguments."
					slice_ids[2].append((w_start, w_end))
					xt_list.append(x[w_start:w_end, :])
					yt_list.append(y[w_end - 1])
					# Iterates over the temporal axis extracting the test data
					w_start, w_end = (w_start - self.window, w_end - self.window)

				xd_list, yd_list = [], []
				# Separating the tuning data right before the test data
				for _ in range(self.tuning_samples):
					assert w_start >= 0, "Invalid arguments."
					slice_ids[1].append((w_start, w_end))
					xd_list.append(x[w_start:w_end, :])
					yd_list.append(y[w_end - 1])
					# Iterates over the temporal axis extracting the test data
					w_start, w_end = (w_start - self.window, w_end - self.window)

				x_list, y_list = [], []
				# Train ends on 0 and starts on w-end
				for idx in range(w_end, self.window - 1, -1):
					w_start, w_end = (idx - self.window, idx)
					assert w_start >= 0, "Invalid arguments."
					slice_ids[0].append((w_start, w_end))
					x_list.append(x[w_start:w_end, :])
					y_list.append(y[w_end - 1])

				# Will join the final dataset
				xt.append(torch.stack(xt_list))
				yt.append(torch.stack(yt_list))
				xd.append(torch.stack(xd_list))
				yd.append(torch.stack(yd_list))
				xf.append(torch.stack(x_list))
				yf.append(torch.stack(y_list))
			except:
				pass  # Iterate through the next trajectory
			# Checking for data leaking by intersecting indices (no leaking means an empty set)
			assert len(set(slice_ids[0]) & (set(slice_ids[1]) | set(slice_ids[2]))) == 0

		# Shared samples as Double Floats
		xt = torch.cat(xt).to(torch.float64)
		yt = torch.cat(yt).to(torch.float64)
		xd = torch.cat(xd).to(torch.float64)
		yd = torch.cat(yd).to(torch.float64)
		x = torch.cat(xf).to(torch.float64)
		y = torch.cat(yf).to(torch.float64)

		if self.normalize_data:
			# Z-Score Normalization
			x_std, x_mean = torch.std_mean(x, dim=0, unbiased=False, keepdim=True)
			x_std[x_std == 0] = 1  # avoid zero division
			x = (x - x_mean) / x_std  # training data
			xd = (xd - x_mean) / x_std  # tuning data
			xt = (xt - x_mean) / x_std  # test data
			# Min-Max Normalization
			x_max = x.amax(dim=0, keepdim=True)
			x_min = x.amin(dim=0, keepdim=True)
			x_max[x_max == 0] = 1  # avoid zero division
			x = (x - x_min) / (x_max - x_min)  # training data
			xd = (xd - x_min) / (x_max - x_min)  # tuning data
			xt = (xt - x_min) / (x_max - x_min)  # test data

		# The batches are CPU-pinned for quicker GPU transfer
		dataloader = DataLoader(dataset=TensorDataset(*[x.view(-1, *x.shape[-2:]), y.view(-1, )]),
		                        generator=generator, worker_init_fn=NetworkPlayground.__worker_init,
		                        num_workers= max(0, min(num_workers, cpu_count())),
		                        batch_size=self.batch_size, shuffle=self.shuffle,
		                        drop_last=drop_last, pin_memory=True)

		# Enforcing the shape of the input tuning and test data on reserved RAM address
		xd, yd = xd.view(-1, *xd.shape[-2:]).pin_memory(), yd.view(-1, ).pin_memory()
		xt, yt = xt.view(-1, *xt.shape[-2:]).pin_memory(), yt.view(-1, ).pin_memory()

		# Train, tuning, and test data folds
		return dataloader, (xd, yd), (xt, yt)

	def __print_details(self, verbose=True):
		# Yielding a preview of the network architecture
		if verbose: print("[I] Network Architecture:\n\n", self, end="\n\n")
		# Creating a table to store details about the layers
		table = PrettyTable(["Modules", "Shape", "Parameters"])
		table.align["Parameters"] = "c"
		table.align["Modules"] = "r"
		table.align["Shape"] = "c"
		# Sorting by the number of parameters
		table.sortby = "Parameters"
		total_params = 0
		# Adding details about layers' shape and parameters
		for name, parameter in self.named_parameters():
			if not parameter.requires_grad: continue
			shape = str(list(parameter.shape))
			shape = shape.replace(", ", " x ")
			shape = shape.replace("]", "")
			shape = shape.replace("[", "")
			n_params = parameter.numel()
			table.add_row([name, shape, n_params])
			total_params += n_params
		table.add_row(["TOTAL", "", total_params])
		# Printing the resulting table before proceeding with training
		if verbose: print(f"[I] Training Details:\n\n{table}\n\n[I] Network Training:", end="")
		return total_params

	@staticmethod
	def __worker_init(worker_id):
		"""
			Enforce reproducibility for PyTorch's Dataloader.
		"""
		worker_seed = torch.initial_seed() % 2 ** 32
		np.random.seed(worker_seed)
		random.seed(worker_seed)
		return worker_id

	@staticmethod
	def __reseeding(random_seed):
		"""
			Avoid losing determinism on Cuda 10.2.
		"""
		torch.backends.cudnn.deterministic = True
		torch.use_deterministic_algorithms(True)
		torch.cuda.manual_seed_all(random_seed)
		torch.manual_seed(random_seed)
		np.random.seed(random_seed)
		random.seed(random_seed)

	def fit(self, x, y):
		"""
			Training routine for time-series binary classification.
			:param x: tensor of shape (n_samples, n_timestamps, n_features)
				A tensor of float-valued data features to use for classification.
			:param y: tensor of shape (n_samples, n_timestamps, n_features)
				A tensor of integer-valued monotonically-increasing labels.
		"""
		# >>> Test nothing
		torcheck.verbose_off()  # Removes extra information
		torcheck.disable()  # Disable the debugging module
		# >>> Test everything
		# torcheck.verbose_on()  # Includes the tensor info
		# torcheck.enable()  # Activate the checking module

		# Forcing Determinism
		generator = torch.Generator()
		generator.manual_seed(self.random_seed)
		NetworkPlayground.__reseeding(self.random_seed)

		# Training Variables
		os.makedirs("./training-checkpoints", exist_ok=True)
		self.epoch, unimprovement, keep_training = 0, 0, True
		checkpoint_path = os.path.join("./.NN-%d-%s-%s.pt" % (self.random_seed, self.hash, self.suffix))

		# Sliced Test Data
		dataloader, (x_dev, y_dev), (x_out, y_out) = self.data_preparation(x, y, generator=generator)
		# Beware of this might overflow the GPU memory depending on the size of the dataset
		x_dev, y_dev, x_out, y_out = x_dev.cuda(), y_dev.cpu(), x_out.cuda(), y_out.cpu()

		while keep_training:
			try:  # Hit CTRL+C to stop training
				f_losses, d_losses, lr_list = [], [], []
				bar_format = "{desc}{percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]{bar:-10b}"
				with tqdm(dataloader, total=len(dataloader), unit="batch", bar_format=bar_format, disable=(not self.verbose), ncols=200) as mini_batches:
					mini_batches.set_description("#%04d" % self.epoch)

					for x_fit, y_fit in mini_batches:
						x_fit, y_fit = x_fit.cuda(), y_fit.cuda()
						_ = self.__fit(x_fit, y_fit)  # Training the Network
						y_fit, yx_fit = y_fit.cpu(), self.predict(x_fit)
						y_dev, yx_dev = y_dev.cpu(), self.predict(x_dev)
						y_out, yx_out = y_out.cpu(), self.predict(x_out)
						d_losses.append(self.BCE(yx_dev, y_dev))

						mini_batches.set_postfix(
								stop="%03d" % unimprovement,
								rate="%08.7f" % self.optimizer.param_groups[0]["lr"],
								a_acc="%05.3f" % balanced_accuracy_score(y_fit, yx_fit),
								a_fsc="%05.3f" % f1_score(y_fit, yx_fit, average="macro", zero_division=1.),
								b_acc="%05.3f" % balanced_accuracy_score(y_dev, yx_dev),
								b_fsc="%05.3f" % f1_score(y_dev, yx_dev, average="macro", zero_division=1.),
								c_acc="%05.3f" % balanced_accuracy_score(y_out, yx_out),
								c_fsc="%05.3f" % f1_score(y_out, yx_out, average="macro", zero_division=1.),
						)

					current_loss = np.array(d_losses).mean(axis=0)
					# LR scheduling using the development loss
					self.scheduler.step(current_loss)

					if min(self.min_loss, current_loss) == current_loss:
						unimprovement = unimprovement + 1 if (self.min_loss - current_loss) < self.improvement_threshold else 0
						self.min_loss = current_loss  # the minimum training loss is the current one
						torch.save({  # saving a snapshot of the current model
							"epoch": self.epoch,
							"details": self.details,
							"min_loss": self.min_loss,
							"model_state_dict": self.state_dict(),
							"optimizer_state_dict": self.optimizer.state_dict(),
						}, checkpoint_path)  # always overwrites the previous one
						if self.verbose and unimprovement == 0:
							print("\n", classification_report(y_out.cpu().numpy(), self.predict(x_out), labels=[0, 1], target_names=["Sailing (0)", "Fishing (1)"], zero_division=1.))
					else:
						unimprovement += 1

					# Stop training when the patience runs over after not seeing improvements
					keep_training = False if unimprovement >= self.learning_patience else True
					self.epoch += 1  # starting a new epoch
			except KeyboardInterrupt:
				keep_training = False

		# Loading the best epoch from the disk
		checkpoint = torch.load(os.path.join("./training-checkpoints/", checkpoint_path))
		pp = pprint.PrettyPrinter(indent=4)

		# Restoring previous states
		self.epoch = checkpoint["epoch"]
		self.details = checkpoint["details"]
		self.min_loss = checkpoint["min_loss"]
		self.load_state_dict(checkpoint["model_state_dict"])
		self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

		pp.pprint(self.details); print(""); self.__print_details()  # report the hyperparameters, number of internal parameters, and test again with the last seen weights
		print("\n", classification_report(y_out.cpu().numpy(), self.predict(x_out), labels=[0, 1], target_names=["Sailing (0)", "Fishing (1)"], zero_division=1.))
		shutil.move(checkpoint_path, os.path.join("./training-checkpoints/", checkpoint_path[3:]))  # move the checkpoint to a permanent folder
		return self.min_loss

	def test_checkpoint(self, x, y, checkpoint_path):
		"""
			Training routine for time-series binary classification.
			:param x: tensor of shape (n_samples, n_timestamps, n_features)
				A tensor of float-valued data features to use for classification.
			:param y: tensor of shape (n_samples, n_timestamps, n_features)
				A tensor of integer-valued monotonically-increasing labels.
			:param checkpoint_path: string
				Path of the trained model to load and perform the experiments.
		"""
		# Forcing Determinism
		generator = torch.Generator()
		generator.manual_seed(self.random_seed)
		NetworkPlayground.__reseeding(self.random_seed)

		# Sliced Test Data
		_, (x_dev, y_dev), (x_out, y_out) = self.data_preparation(x, y, generator=generator)
		x_dev, y_dev, x_out, y_out = x_dev.cpu(), y_dev.cpu().numpy(), x_out.cpu(), y_out.cpu().numpy()

		# Loading the best epoch from the disk
		checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

		# Restoring previous states
		self.epoch = checkpoint["epoch"]
		self.details = checkpoint["details"]
		self.min_loss = checkpoint["min_loss"]
		self.load_state_dict(checkpoint["model_state_dict"])
		self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
		self.cpu()  # Move to the CPU device
		for key, value in self.details.items():
			# Mapping all kwargs to attributes
			setattr(self, key, value)

		yp_dev = self.predict(x_dev).cpu().numpy()
		yp_out = self.predict(x_out).cpu().numpy()    

		parameters = self.__print_details(verbose=False)
		return (y_dev, yp_dev), (y_out, yp_out), parameters
