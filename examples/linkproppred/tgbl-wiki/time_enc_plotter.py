import math
import timeit
import random
import os
import os.path as osp
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# internal imports
from tgb.utils.utils import get_args
from modules.time_enc import get_time_encoder


@torch.no_grad()
def eval_time_func(t0, t1, nsteps, time_enc, time_dim, batch_size=1024):
    ts = torch.linspace(t0, t1, nsteps).to(device)
    outs = torch.zeros((nsteps, time_dim), device=device)
    for start in tqdm(range(0, nsteps, batch_size)):
        end = min(start + batch_size, nsteps)
        outs[start: end] = time_enc(ts[start: end])
    return ts.cpu(), outs.cpu()


sns.set_style("whitegrid")


args, _ = get_args()
DATA = "tgbl-wiki"
MODEL_NAME = "TGN"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

te = ["graph_mixer", "fixed_gaussian", "decay_amp_gm"]
tes = [get_time_encoder(i, 100).to(device) for i in te]
names = [r"cos", r"$\phi_{N}$", r"$\phi_{decay}$"]
alphas = [0.4, 1, 1]

idx = [7, 7, 7]

centre = 80
tes[1].lin.bias.data = tes[1].lin.weight.data[:, 0] * -centre
tes[0].lin.bias.data = tes[1].lin.bias.data
tes[2].lin.bias.data = tes[1].lin.bias.data

tes[2].power.data[:] = 0.6

time_min = 0
time_max = 2 * centre

num_time_samples = 5000

for n in range(100):
    # 5:3 before
    fig, ax = plt.subplots(figsize=(4, 3))

    for i in range(len(names)):
        ts, out = eval_time_func(time_min, time_max, num_time_samples, tes[i], args.time_dim)
        ax.plot(ts, out[:, idx[i]+n], color=f"C{i}", label=names[i], alpha=alphas[i])

        print("Bias =", tes[i].lin.bias[idx[i]+n])

    ax.set_ylim((-1.5, 1.5))

    ax.legend(loc='upper right')
    ax.set_xlabel("t")

    # legend = ax.legend(loc='upper right', fontsize='medium')
    # legend.set_title('Legend', prop={'size': 8})  # Set legend title and adjust its size

    plt.tight_layout()
    plt.savefig("decay_gaussian_example2.pdf", bbox_inches='tight', pad_inches=0.1)

    # plt.savefig("output.pdf")
    plt.show()
