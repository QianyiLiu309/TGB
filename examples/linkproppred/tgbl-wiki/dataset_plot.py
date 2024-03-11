"""
Dynamic Link Prediction with a TGN model with Early Stopping
Reference: 
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py

command for an example run:
    python examples/linkproppred/tgbl-wiki/tgn.py --data "tgbl-wiki" --num_run 1 --seed 1
"""

import sys

sys.path.append("..")

import numpy as np
import torch
from torch_geometric.loader import TemporalDataLoader
from tgb.utils.utils import get_args, set_random_seed, save_results
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from plot_utils import get_temporal_edge_times
from sklearn.neighbors import KernelDensity
import math


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
# val_data = data[val_mask]
# test_data = data[test_mask]

test_pairs = set(zip(train_data.src.tolist(), train_data.dst.tolist()))

all_gaps = []

for i, j in test_pairs:
    inds = np.logical_and(train_data.src == i, train_data.dst == j)
    ts = train_data.t[inds]
    all_gaps.append(ts[1:] - ts[:-1])

all_gaps = np.concatenate(all_gaps)
all_gaps = np.sort(all_gaps)


plt.hist(all_gaps, bins=1000)
plt.yscale('log')
plt.show()

print(f"Saving dataset data to ../dataset_stats/{args.data}_gaps.npy")
np.save(f"../dataset_stats/{args.data}_gaps.npy", all_gaps)

# counts = {i: test_pairs.count(i) for i in set(test_pairs)}
# biggest = sorted(set(test_pairs), key=lambda i: counts[i], reverse=True)


## DFT
print("DFT Time")
data_range = np.max(all_gaps) - np.min(all_gaps)
# Fit PDF to data
kde = KernelDensity(bandwidth=data_range/100)
kde.fit(all_gaps[:, None])
num_samples = 5000
sample_spacing = data_range / num_samples
# Sample PDF at linearly separated points
x = np.linspace(min(all_gaps), max(all_gaps), num_samples)[:, None]
log_dens = kde.score_samples(x)
# Plot approximated PDF
plt.fill_between(x[:, 0], np.exp(log_dens))
plt.yscale('log')
plt.show()
# Apply FFT
signal = log_dens  # working with log leads to less numerical errors but same output
n = signal.shape[0]
fft = np.fft.fft(signal)[:n//2]  # n//2 to discard negative frequencies
# Get top frequencies
amplitudes = np.abs(fft)
freqs = np.fft.fftfreq(n, d=sample_spacing)[:n//2]  # corresponding freqs
F = freqs[np.argsort(-amplitudes)]  # freqs sorted by increasing amplitude

weights = 2 * math.pi * F

print(f"Top periods in hours:")
print((1 / F[:100]) / 60 / 60)
print("Top 100 weights:")
print(weights[:100])

print(f"Saving dataset data to ../dataset_stats/{args.data}_frequencies_weight.npy")
np.save(f"../dataset_stats/{args.data}_frequencies_weight.npy", weights)

