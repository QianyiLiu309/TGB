"""
Time Encoding Module

Reference:
    - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/tgn.html
"""


import torch
from torch import Tensor
from torch.nn import Linear


class TimeEncoder(torch.nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, t: Tensor) -> Tensor:
        return self.lin(t.view(-1, 1)).cos()


class ExpTimeEncoder(torch.nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels, bias=False)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, t: Tensor) -> Tensor:
        # [w1 t, w2 t, w3 t, ...]
        xs = self.lin(t.view(-1, 1)).abs()
        return torch.exp(-xs)


class GaussianTimeEncoder(torch.nn.Module):
    """Inspired by Gaussian PDF"""
    def __init__(self, out_channels: int, eps=1e-4):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels, bias=False)
        self.lin_s = Linear(1, out_channels, bias=False)
        self.eps = eps
        with torch.no_grad():
            self.lin_s.weight *= 10

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.lin_s.reset_parameters()

    def forward(self, t: Tensor) -> Tensor:
        t = t.view(-1, 1)
        m = self.lin(t)
        s = self.lin_s(t)
        return torch.exp(-((t - m) / (s + self.eps)) ** 2)
