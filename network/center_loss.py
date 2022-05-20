# coding=utf-8
#
#  Copyright 2022, Gabriel Spadon, all rights reserved.
#  This code is under GNU General Public License v3.0.
#      www.spadon.com.br & gabriel@spadon.com.br
#
# This script requires setting "CUBLAS_WORKSPACE_CONFIG=:16:8" as an environment variable.

import torch
import numpy as np
from torch import device


class CenterLoss(torch.nn.Module):
    """
    Center Loss for classification tasks, adapted for a three-dimensional input using different reduction functions
        [*] Wen et al., "A Discriminative Feature Learning Approach for Deep Face Recognition", ECCV (2016)
    """

    def __init__(self, classes=2, variables=4, reduction="mean-log"):
        """
        Initializes the trainable parameters of the loss function
        :param classes: integer (default: 2)
            The number of target classes in the classification task.
        :param variables: integer (default: 4)
            The number of variables within the multivariate dataset.
        :param reduction: string (default: mean-log)
            The way o reduce the list of losses when in a three-dimensional task.
        """
        super(CenterLoss, self).__init__()

        self.centers = torch.randn(classes, variables).cuda()
        self.centers = torch.nn.Parameter(self.centers)
        self.reduction = reduction.lower()
        self.classes = classes

    def forward(self, x, y):
        """
        Args:
            x: feature matrix with shape (batch_size, [time,] feat_dim).
            y: ground truth labels with shape (batch_size).
        """
        if len(x.shape) < 2 or len(x.shape) > 3:
            raise ValueError("Not implemented.")
        elif len(x.shape) == 2:
            torch.unsqueeze(x, 1)

        losses = []
        for t in range(0, x.shape[1]):
            x_input = torch.squeeze(x[:, t, :])
            batch_size = x_input.shape[0]  # first dimension of tensor
            dist_mtx_1 = torch.pow(x_input, 2).sum(dim=1, keepdim=True)
            dist_mtx_1 = dist_mtx_1.expand(batch_size, self.classes)
            dist_mtx_2 = torch.pow(self.centers, 2).sum(dim=1, keepdim=True)
            dist_mtx_2 = dist_mtx_2.expand(self.classes, batch_size).T
            _dist_mtx_ = (dist_mtx_1 + dist_mtx_2).addmm(x_input, self.centers.T, beta=1, alpha=-2)

            y = y.unsqueeze(1) if len(y.shape) == 1 else y
            classes = torch.arange(self.classes).to(device="cuda", dtype=torch.float64)
            mask = y.expand(batch_size, self.classes).eq(classes.expand(batch_size, self.classes))
            losses.append((_dist_mtx_ * mask.to(torch.float64)).clamp(min=1e-12, max=1e+12).sum() / batch_size)

        if self.reduction == "sum-log":
            return torch.sum(torch.log1p(torch.stack(losses)))
        elif self.reduction == "median-log":
            return torch.median(torch.log1p(torch.stack(losses)))
        elif self.reduction == "mean-log":
            return torch.mean(torch.log1p(torch.stack(losses)))
        if self.reduction == "sum":
            return torch.sum(torch.stack(losses))
        elif self.reduction == "median":
            return torch.median(torch.stack(losses))
        # Default reduction is the mean/average
        return torch.mean(torch.stack(losses))
