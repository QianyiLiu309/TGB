"""
Time Encoding Module

Reference:
    - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/tgn.html
"""

import torch
from torch import Tensor
from torch.nn import Linear, Parameter

import numpy as np


class TimeEncoder(torch.nn.Module):
    def __init__(self, out_channels: int, mul=1):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels)
        self.mul = mul

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, t: Tensor) -> Tensor:
        t = t * self.mul
        return self.lin(t.view(-1, 1)).cos()


class ExpTimeEncoder(torch.nn.Module):
    def __init__(self, out_channels: int, mul=1):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels, bias=False)
        self.mul = mul

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, t: Tensor) -> Tensor:
        t = t * self.mul
        # [w1 t, w2 t, w3 t, ...]
        xs = self.lin(t.view(-1, 1)).abs()
        return torch.exp(-xs)


class GaussianTimeEncoder(torch.nn.Module):
    """Inspired by Gaussian PDF"""

    def __init__(self, out_channels: int, mul=1):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels, bias=True)
        self.mul = mul

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, t: Tensor) -> Tensor:
        t = t * self.mul
        return torch.exp(-self.lin(t.view(-1, 1)) ** 2)


class TimeEncoderGM(torch.nn.Module):

    def __init__(self, out_channels: int, parameter_requires_grad: bool = True):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super().__init__()

        self.out_channels = out_channels
        # trainable parameters for time encoding
        self.lin = Linear(1, out_channels)
        self.lin.weight = Parameter(
            (
                torch.from_numpy(
                    1 / 10 ** np.linspace(0, 9, out_channels, dtype=np.float32)
                )
            ).reshape(out_channels, -1)
        )
        self.lin.bias = Parameter(torch.zeros(out_channels))

        if not parameter_requires_grad:
            self.lin.weight.requires_grad = False
            self.lin.bias.requires_grad = False

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, timestamps: torch.Tensor):
        """
        compute time encodings of time in timestamps
        :param timestamps: Tensor, shape (batch_size, seq_len)
        :return:
        """
        # Tensor, shape (batch_size, seq_len, 1)
        timestamps = timestamps.view(-1, 1)

        # Tensor, shape (batch_size, seq_len, time_dim)
        output = torch.cos(self.lin(timestamps))

        return output


class PartiallyLearnedTimeEncoder(torch.nn.Module):
    def __init__(self, out_channels: int):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super().__init__()

        self.out_channels = out_channels
        # trainable parameters for time encoding
        self.frequencies = Parameter(
            torch.from_numpy(
                1 / 10 ** np.linspace(-2, 7, out_channels, dtype=np.float32)
            ).unsqueeze(0)
        )
        self.frequencies.requires_grad = False

        self.lin = Linear(1, out_channels, bias=True)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, timestamps: torch.Tensor):
        """
        compute time encodings of time in timestamps
        :param timestamps: Tensor, shape (batch_size, seq_len)
        :return:
        """
        # Tensor, shape (batch_size, seq_len, 1)
        timestamps = timestamps.view(-1, 1)

        output = torch.matmul(timestamps, self.lin.weight.t())
        if output.shape[0] != 0:
            output = output * self.frequencies
        output = output + self.lin.bias
        output = torch.cos(output)

        return output

    def get_parameter_norm(self):
        return torch.norm(self.lin.weight, p=2) + torch.norm(self.lin.bias, p=2)
