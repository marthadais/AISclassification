import torch
import numpy as np
import multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import RNN, GRU, LSTM, TransformerEncoderLayer


class MyNetwork(torch.nn.Module):
    """
    A Vanilla Neural Network for baselines computation.
    """
    def __init__(self, window, development, horizon, recurrent_unit, hidden_size, recurrent_layers, bidirectional, bias, **kwargs):
        """
        Initializes the layers of the network with the user-input arguments.
        :param window: integer
            The number of time steps to look at in the past.
        :param development: integer
            The number of time-steps reserved for hyperparameter tuning.
        :param horizon: integer
            The number of time steps to predict in the future.
        :param recurrent_unit: string
            Defines the architecture used as a recurrent layer.
        :param hidden_size: integer
            The number of elements in the hidden unit of the RNN.
        :param recurrent_layers: string
            The number of stacked recurrent unities.
        :param bidirectional: boolean
            Sets the recurrent unit as bidirectional if set to True.
        :param bias: boolean
            Sets bias vectors permanently to zeros if set to False.
        """
        super(MyNetwork, self).__init__()

        # Attributes
        self.bias = bias
        self.window = window
        self.horizon = horizon
        self.development = development
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.recurrent_unit = recurrent_unit
        self.recurrent_layers = recurrent_layers

        # Evaluation Metrics
        self.L1Loss = torch.nn.L1Loss
        self.MSELoss = torch.nn.MSELoss
        self.HuberLoss = torch.nn.HuberLoss

        # Training Attributes
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        for key, value in kwargs.items():
            # Mapping all kwargs to attributes
            setattr(self, key, value)

        # Neural Network Layers
        self.RNN = eval(recurrent_unit)(
                input_size=window, hidden_size=hidden_size,
                num_layers=recurrent_layers, batch_first=True,
                bidirectional=bidirectional, bias=bias
        )

        self.Linear = torch.nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x):
        """
        Defines the computation performed at every call.
        :param x: array-like of shape (samples, window, variables)
            Observations from the past window-sized time-steps.
        :return: array-like of shape (samples, horizon, variables)
            Predictions for the next horizon-sized time-steps.
        """
        x, _ = self.RNN(x.permute(0, 2, 1))  # time to the last axis
        x = x.permute(0, 2, 1)  # variable to the last axis
        if self.bidirectional:
            x = x.view(-1, self.horizon, 2, self.variables)
            x = x.sum(axis=2)  # merging branches by sum
        x = self.Linear(x)
        return x

    def __train(self, x, y, clipping_value=1.):
        """
        PyTorch training routine.
        :param x: array-like of shape (samples, window, variables)
            Observations from the past window-sized time-steps.
        :param y: array-like of shape (samples, horizon, variables)
            Predictions for the next horizon-sized time-steps.
        :param clipping_value: float
            The max norm of the gradients.
        :return: array-like of shape (1,)
            The criterion loss.
        """
        self.train()
        self.optimizer.zero_grad()

        # Forward propagation
        y_pred = self.forward(x)

        # Rollback any normalization-like
        y_pred, y = self.roll_back(y_pred), y

        # Computing the resulting loss
        loss = self.criterion(y_pred, y)

        loss.backward()
        # Gradient value clipping
        torch.nn.utils.clip_grad_norm(self.parameters(), clipping_value)
        self.optimizer.step()  # Updating parameters

        return loss.detach().cpu().numpy()

    def __test(self, x, y):
        """
        PyTorch testing routine.
        :param x: array-like of shape (samples, window, variables)
            Observations from the past window-sized time-steps.
        :param y: array-like of shape (samples, horizon, variables)
            Predictions for the next horizon-sized time-steps.
        :return: array-like of shape (3,)
            A set of three evaluation metrics.
        """
        self.eval()
        with torch.no_grad():
            y_pred = self.forward(x)

        # Rollback any normalization-like
        y_pred, y = self.roll_back(y_pred), y

        # The batch criterion loss
        return np.array([
            self.L1Loss(y_pred, y).cpu().numpy(),  # Mean Absolute Error (MAE)
            self.MSELoss(y_pred, y).cpu().numpy(),  # Mean Squared Error (MSE)
            self.HuberLoss(y_pred, y).cpu().numpy(),  # Huber Loss (HL)
        ])

    def data_preparation(self, x, y, batch_size=32, shuffle=True, normalize_data=True, num_workers=1, drop_last=True):
        """
        Iterates over the time axis, segmenting the time series for neural network training.
        :param x: tensor of shape (n_samples, n_timestamps, n_features)
            A tensor of float-valued data features to use for classification.
        :param y: tensor of shape (n_samples, n_timestamps, n_features)
            A tensor of integer-valued monotonically-increasing labels.
        :param batch_size: integer
            How many samples per batch to load.
        :param shuffle: boolean
            Set to True to have the data reshuffled at every epoch.
        :param normalize_data: boolean
            If True applies z-score followed by min-max normalization on the features.
        :param num_workers: integer
            How many subprocesses to use for data loading (set to -1 for all).
        :param drop_last: boolean
            Set to True to drop the last batch if incomplete.
        """
        slice_ids = [[], [], []]
        xt_list, yt_list = [], []
        # Isolating the test data from the end of the dataset
        w_start, w_end = (x.shape[1] - self.window, x.shape[1])
        for _ in range(self.horizon):  # Samples reserved for testing
            assert w_start >= 0, "Invalid arguments."
            slice_ids[2].append((w_start, w_end))
            xt_list.append(x[:, w_start:w_end, :])
            yt_list.append(y[:, w_end - 1])
            # Iterates over the temporal axis extracting the test data
            w_start, w_end = (w_start - self.window, w_end - self.window)
        xt, yt = torch.stack(xt_list), torch.stack(yt_list)

        xd_list, yd_list = [], []
        # Separating the development data right before the test data
        for _ in range(self.development):  # Samples reserved for development
            assert w_start >= 0, "Invalid arguments."
            slice_ids[1].append((w_start, w_end))
            xd_list.append(x[:, w_start:w_end, :])
            yd_list.append(y[:, w_end - 1])
            # Iterates over the temporal axis extracting the test data
            w_start, w_end = (w_start - self.window, w_end - self.window)
        xd, yd = torch.stack(xd_list), torch.stack(yd_list)

        x_list, y_list = [], []
        # Train ends on 0 and starts on w-end
        for idx in range(w_end, self.window - 1, -1):
            w_start, w_end = (idx - self.window, idx)
            assert w_start >= 0, "Invalid arguments."
            slice_ids[0].append((w_start, w_end))
            x_list.append(x[:, w_start:w_end, :])
            y_list.append(y[:, w_end - 1])
        x, y = torch.stack(x_list), torch.stack(y_list)

        # Checking for data leaking by intersecting indices (no leaking means an empty set)
        assert len(set(slice_ids[0]) & (set(slice_ids[1]) | set(slice_ids[2]))) == 0

        if normalize_data:
            # Z-Score Normalization
            x_std, x_mean = torch.std_mean(x, dim=(0, 2), unbiased=False, keepdim=True)
            x_std[x_std == 0] = 1  # Avoid zero division
            x = (x - x_mean) / x_std  # Training Data
            xd = (xd - x_mean) / x_std  # Tuning Data
            xt = (xt - x_mean) / x_std  # Test Data
            # Min-Max Normalization
            x_max = x.amax(dim=(0, 2), keepdim=True)
            x_min = x.amin(dim=(0, 2), keepdim=True)
            x_max[x_max == 0] = 1  # Avoid zero division
            x = (x - x_min) / (x_max - x_min)  # Training Data
            xd = (xd - x_min) / (x_max - x_min)  # Tuning Data
            xt = (xt - x_min) / (x_max - x_min)  # Test Data

        # The batches are CPU-pinned for quicker GPU transfer
        dataloader = DataLoader(dataset=TensorDataset(*[x.view(-1, *x.shape[-2:]), y.view(-1,)]),
                                num_workers=max(0, min(num_workers, mp.cpu_count())),
                                batch_size=batch_size, shuffle=shuffle,
                                drop_last=drop_last, pin_memory=True)

        # Enforcing the shape of the input tuning and test data on reserved RAM address
        xd, yd = xd.view(-1, *xd.shape[-2:]).pin_memory(), yd.view(-1,).pin_memory()
        xt, yt = xt.view(-1, *xt.shape[-2:]).pin_memory(), yt.view(-1,).pin_memory()

        # Train, Tuning, and Test data folds
        return dataloader, [(xd, yd), (xt, yt)]
