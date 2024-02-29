"""
Time Encoding Module

Reference:
    - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/tgn.html
"""


import torch
from torch import Tensor
from torch.nn import Linear


class TimeEncoder(torch.nn.Module):
    def __init__(self, out_channels: int, mul=0.01):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels)
        self.mul = mul
        with torch.no_grad():
            self.lin.weight *= mul

    def reset_parameters(self):
        self.lin.reset_parameters()
        with torch.no_grad():
            self.lin.weight *= self.mul

    def forward(self, t: Tensor) -> Tensor:
        return self.lin(t.view(-1, 1)).cos()


class ExpTimeEncoder(torch.nn.Module):
    def __init__(self, out_channels: int, mul=0.01):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels, bias=False)
        self.mul = mul
        with torch.no_grad():
            self.lin.weight *= mul

    def reset_parameters(self):
        self.lin.reset_parameters()
        with torch.no_grad():
            self.lin.weight *= self.mul

    def forward(self, t: Tensor) -> Tensor:
        # [w1 t, w2 t, w3 t, ...]
        xs = self.lin(t.view(-1, 1)).abs()
        return torch.exp(-xs)


class GaussianTimeEncoder(torch.nn.Module):
    """Inspired by Gaussian PDF"""
    def __init__(self, out_channels: int, mul=0.01):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels, bias=True)
        self.mul = mul
        with torch.no_grad():
            self.lin.weight *= mul

    def reset_parameters(self):
        self.lin.reset_parameters()
        with torch.no_grad():
            self.lin.weight *= self.mul

    def forward(self, t: Tensor) -> Tensor:
        return torch.exp(-self.lin(t.view(-1, 1)) ** 2)
