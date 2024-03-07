"""Evaluate how much parameters have changed before and after training"""
"""
Dynamic Link Prediction with a TGN model with Early Stopping
Reference: 
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py

command for an example run:
    python examples/linkproppred/tgbl-wiki/tgn.py --data "tgbl-wiki" --num_run 1 --seed 1
"""

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

# internal imports
from tgb.utils.utils import get_args, set_random_seed, save_results
from tgb.linkproppred.evaluate import Evaluator
from modules.decoder import LinkPredictor
from modules.emb_module import GraphAttentionEmbedding
from modules.msg_func import IdentityMessage
from modules.msg_agg import LastAggregator
from modules.neighbor_loader import LastNeighborLoader
from modules.memory_module import TGNMemory
from modules.early_stopping import EarlyStopMonitor
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset


def extract_params(model):
    d = {}
    for key in model:
        module = model[key]
        ps = {(key+"."+name): param.detach().cpu().clone() for name, param in module.named_parameters()}
        d.update(ps)
    return d


@torch.no_grad()
def eval_time_func(t0, t1, nsteps, time_enc, time_dim, batch_size=1024):
    ts = torch.linspace(t0, t1, nsteps).to(device)
    outs = torch.zeros((nsteps, time_dim), device=device)
    for start in tqdm(range(0, nsteps, batch_size)):
        end = min(start + batch_size, nsteps)
        outs[start: end] = time_enc(ts[start: end])
    return ts.cpu(), outs.cpu()


args, _ = get_args()
DATA = "tgbl-wiki"
MODEL_NAME = "TGN"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_diff = []
all_outs_before = []
all_outs_after = []

for run_idx in range(args.num_run):
    torch.manual_seed(run_idx + args.seed)
    set_random_seed(run_idx + args.seed)

    dataset = PyGLinkPropPredDataset(name=DATA, root="datasets")
    data = dataset.get_TemporalData()

    memory = TGNMemory(
        data.num_nodes,
        data.msg.size(-1),
        args.mem_dim,
        args.time_dim,
        message_module=IdentityMessage(data.msg.size(-1), args.mem_dim, args.time_dim),
        aggregator_module=LastAggregator(),
        time_encoder=args.time_encoder,
        multiplier=args.mul,
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=args.mem_dim,
        out_channels=args.emb_dim,
        msg_dim=data.msg.size(-1),
        time_enc=memory.time_enc,
    ).to(device)

    link_pred = LinkPredictor(in_channels=args.emb_dim).to(device)

    model = {"memory": memory, "gnn": gnn, "link_pred": link_pred}
    # print(model)

    results_path = f"{osp.dirname(osp.abspath(__file__))}/saved_results"
    if not osp.exists(results_path):
        os.mkdir(results_path)
        print("INFO: Create directory {}".format(results_path))
    Path(results_path).mkdir(parents=True, exist_ok=True)
    results_filename = f"{results_path}/{MODEL_NAME}_{DATA}_results.json"
    save_model_dir = f"{osp.dirname(osp.abspath(__file__))}/saved_models/"
    save_model_id = f"{MODEL_NAME}_{DATA}_{args.seed}_{run_idx}_{args.time_encoder}_{args.mul}"
    early_stopper = EarlyStopMonitor(
        save_model_dir=save_model_dir,
        save_model_id=save_model_id,
        tolerance=args.tolerance,
        patience=args.patience,
    )

    num_time_samples = 5000

    init_params = extract_params(model)

    # ts, outs_before = eval_time_func(data.t[0], data.t[-1], num_time_samples, memory.time_enc, args.time_dim)
    # all_outs_before.append(outs_before)

    ########### Load checkpoint
    early_stopper.load_checkpoint(model)
    trained_params = extract_params(model)

    # ts, outs_after = eval_time_func(data.t[0], data.t[-1], num_time_samples, memory.time_enc, args.time_dim)
    # all_outs_after.append(outs_after)

    # Plot some of the time components
    # for i in range(0, args.time_dim, 10):
    #     plt.plot(ts, outs_before[:, i], color="C0", label="Before")
    #     plt.plot(ts, outs_after[:, i], color="C1", label="After")
    #     plt.legend()
    #     plt.xlabel("t")
    #     plt.show()

    diffs = {}

    pnames = list(init_params.keys())

    for name in pnames:
        old = init_params[name]
        new = trained_params[name]

        # Square diff
        # d = torch.mean((old - new)**2).item()

        # Abs diff
        d = torch.mean((old - new).abs()).item()

        # mean multiplier
        # inds = (old != 0)
        # d = torch.mean(new[inds] / old[inds])

        diffs[name] = d

    # for name in sorted(diffs, key=lambda i:diffs[i], reverse=True):
    #     print(f"{diffs[name]:.6f} \t\t {name}")

    all_diff.append(diffs)

overall_diff = {}
for name in pnames:
    xs = np.array([d[name] for d in all_diff])
    overall_diff[name] = (xs.mean().item(), xs.std().item())

for name in sorted(overall_diff, key=lambda i: overall_diff[i][0], reverse=True):
    m, s = overall_diff[name]
    print(f"{m:.7f} +- {s:.7f} \t\t {name}")
