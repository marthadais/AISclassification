import torch
import numpy as np


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
