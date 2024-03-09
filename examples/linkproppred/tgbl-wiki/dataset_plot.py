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

from plot_utils import get_temporal_edge_times


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
val_data = data[val_mask]
test_data = data[test_mask]

test_pairs = set(zip(test_data.src.tolist(), test_data.dst.tolist()))

all_gaps = []

for i, j in test_pairs:
    inds = np.logical_and(test_data.src == i, test_data.dst == j)
    ts = test_data.t[inds]
    all_gaps.append(ts[1:] - ts[:-1])

all_gaps = np.concatenate(all_gaps)
all_gaps = np.sort(all_gaps)


plt.hist(all_gaps, bins=1000)
plt.yscale('log')
plt.show()

# counts = {i: test_pairs.count(i) for i in set(test_pairs)}
# biggest = sorted(set(test_pairs), key=lambda i: counts[i], reverse=True)

