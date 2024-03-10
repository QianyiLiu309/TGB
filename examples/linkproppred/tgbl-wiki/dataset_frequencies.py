"""
Dynamic Link Prediction with a TGN model with Early Stopping
Reference: 
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py

command for an example run:
    python examples/linkproppred/tgbl-wiki/tgn.py --data "tgbl-wiki" --num_run 1 --seed 1
"""

import sys

sys.path.append("..")

import timeit

import os
import os.path as osp
from pathlib import Path
import numpy as np

import torch

from torch_geometric.loader import TemporalDataLoader

# internal imports
from tgb.utils.utils import get_args, set_random_seed, save_results
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

from matplotlib import pyplot as plt
from tqdm import tqdm

from scipy.fft import fft


args, _ = get_args()
MODEL_NAME = "TGN"
device = torch.device("cpu")

# data loading
dataset = PyGLinkPropPredDataset(name=args.data, root="datasets")
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask
data = dataset.get_TemporalData()

train_data = data[train_mask]

time_series = train_data.t.numpy()


# Apply FFT to the time series
fft_result = fft(time_series)

# Calculate the frequencies corresponding to the FFT result
n = len(time_series)
frequencies = np.fft.fftfreq(n)

# Keep only positive frequencies (excluding the negative frequencies)
positive_frequencies = frequencies[: n // 2]
positive_fft_result = 2.0 / n * np.abs(fft_result[: n // 2])

# Sort frequencies by magnitude to get the top 100 dominant frequencies
sorted_indices = np.argsort(positive_fft_result)[::-1]
top_100_indices = sorted_indices[0:100]

# Extract the corresponding frequencies and magnitudes
top_100_frequencies = positive_frequencies[top_100_indices]
top_100_magnitudes = positive_fft_result[top_100_indices]

print(f"Top 100 frequencies: {top_100_frequencies}")
print(f"Top 100 magnitudes: {top_100_magnitudes}")

w = 2 * np.pi * top_100_frequencies

print(np.float32(w).dtype)

np.save(f"../dataset_stats/{args.data}_frequencies_weight.npy", np.float32(w))
