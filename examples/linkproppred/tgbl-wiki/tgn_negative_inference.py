"""
Dynamic Link Prediction with a TGN model with Early Stopping
Reference: 
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py

command for an example run:
    python examples/linkproppred/tgbl-wiki/tgn.py --data "tgbl-wiki" --num_run 1 --seed 1
"""

import sys

sys.path.append("..")

import math
import timeit

import os
import os.path as osp
from pathlib import Path
import numpy as np

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear

from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader

from torch_geometric.nn import TransformerConv

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

from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm

from plot_utils import (
    get_temporal_edge_times,
    calculate_average_step_difference,
    calculate_average_step_difference_full_range,
    total_variation_per_unit_time,
)

# ==========
# ========== Define helper function...
# ==========


@torch.no_grad()
def process_split(loader):
    """Just adds the split to memory and neighbour_loader. Shared between train() and val()."""
    for pos_batch in tqdm(loader):
        pos_src, pos_dst, pos_t, pos_msg = (
            pos_batch.src,
            pos_batch.dst,
            pos_batch.t,
            pos_batch.msg,
        )

        # Update memory and neighbor loader with ground-truth state.
        model["memory"].update_state(pos_src, pos_dst, pos_t, pos_msg)
        neighbor_loader.insert(pos_src, pos_dst)
        model["memory"].detach()


@torch.no_grad()
def real_test(loader, neg_sampler, split_mode):
    r"""
    Evaluated the dynamic link prediction
    Evaluation happens as 'one vs. many', meaning that each positive edge is evaluated against many negative edges

    Parameters:
        loader: an object containing positive attributes of the positive edges of the evaluation set
        neg_sampler: an object that gives the negative edges corresponding to each positive edge
        split_mode: specifies whether it is the 'validation' or 'test' set to correctly load the negatives
    Returns:
        perf_metric: the result of the performance evaluaiton
    """
    model["memory"].eval()
    model["gnn"].eval()
    model["link_pred"].eval()

    perf_list = []

    for pos_batch in tqdm(loader):
        pos_src, pos_dst, pos_t, pos_msg = (
            pos_batch.src,
            pos_batch.dst,
            pos_batch.t,
            pos_batch.msg,
        )

        neg_batch_list = neg_sampler.query_batch(
            pos_src, pos_dst, pos_t, split_mode=split_mode
        )

        for idx, neg_batch in enumerate(neg_batch_list):
            src = torch.full((1 + len(neg_batch),), pos_src[idx], device=device)
            dst = torch.tensor(
                np.concatenate(
                    ([np.array([pos_dst.cpu().numpy()[idx]]), np.array(neg_batch)]),
                    axis=0,
                ),
                device=device,
            )

            n_id = torch.cat([src, dst]).unique()
            n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            # Get updated memory of all nodes involved in the computation.
            z, last_update = model["memory"](n_id)
            z = model["gnn"](
                z,
                last_update,
                edge_index,
                data.t[e_id].to(device),
                data.msg[e_id].to(device),
            )

            y_pred = model["link_pred"](z[assoc[src]], z[assoc[dst]])

            # compute MRR
            input_dict = {
                "y_pred_pos": np.array([y_pred[0, :].squeeze(dim=-1).cpu()]),
                "y_pred_neg": np.array(y_pred[1:, :].squeeze(dim=-1).cpu()),
                "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])

        # Update memory and neighbor loader with ground-truth state.
        model["memory"].update_state(pos_src, pos_dst, pos_t, pos_msg)
        neighbor_loader.insert(pos_src, pos_dst)

    perf_metrics = float(torch.tensor(perf_list).mean())

    return perf_metrics


@torch.no_grad()
def test(
    loader,
    target_src,
    target_dst,
    additional_negative_edges,
):
    r"""
    Evaluated the dynamic link prediction
    Evaluation happens as 'one vs. many', meaning that each positive edge is evaluated against many negative edges

    Parameters:
        loader: an object containing positive attributes of the positive edges of the evaluation set
        neg_sampler: an object that gives the negative edges corresponding to each positive edge
        split_mode: specifies whether it is the 'validation' or 'test' set to correctly load the negatives
    Returns:
        perf_metric: the result of the performance evaluaiton
    """

    def evaluate_negative(end_time):
        src_n_ls = []
        dst_n_ls = []
        t_neg_ls = []
        while (
            len(additional_negative_edges) > 0
            and additional_negative_edges[0][2] <= end_time
        ):
            src_n, dst_n, t_neg = additional_negative_edges.pop(0)
            src_n_ls.append(src_n)
            dst_n_ls.append(dst_n)
            t_neg_ls.append(t_neg)

        t_neg = torch.tensor(t_neg_ls, device=device)
        src_n = torch.tensor(src_n_ls, device=device)
        dst_n = torch.tensor(dst_n_ls, device=device)

        n_id = torch.cat(
            [src_n, dst_n],
        ).unique()
        # print(f"n_id: {n_id}")
        if len(n_id) != 0:
            n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            # Get updated memory of all nodes involved in the computation.
            s, last_update = model["memory"](n_id)
            # print(f"z: {z.shape}, last_update: {last_update.shape}")

            y_pred_negs = []
            for sn, dn, tn in zip(src_n, dst_n, t_neg):
                current_time = torch.tensor(
                    [tn] * e_id.shape[0], dtype=torch.float32, device=s.device
                )
                z = model["gnn"](
                    s,
                    last_update,
                    edge_index,
                    data.t[e_id].to(device),
                    data.msg[e_id].to(device),
                    current_time=current_time,
                )

                y_pred_neg = model["link_pred"](z[assoc[sn[None]]], z[assoc[dn[None]]])
                y_pred_negs.append(y_pred_neg)

            y_pred_neg = torch.cat(y_pred_negs)
            predictions_neg.extend(y_pred_neg.squeeze(-1).cpu().numpy())
            timestamps_neg.extend(t_neg.cpu().numpy())

    print(f"Target source: {target_src}, Target destination: {target_dst}")

    model["memory"].eval()
    model["gnn"].eval()
    model["link_pred"].eval()

    predictions = []
    timestamps = []

    predictions_neg = []
    timestamps_neg = []

    adjacent_event_timestamps = []

    relevant_edge_indices = set()

    for pos_batch in tqdm(loader):
        pos_src, pos_dst, pos_t, pos_msg = (
            pos_batch.src,
            pos_batch.dst,
            pos_batch.t,
            pos_batch.msg,
        )

        for idx, _ in enumerate(pos_batch):
            # print(f"Positive src: {pos_src[idx]}, Positive dst: {pos_dst[idx]}")
            src = torch.full((1,), pos_src[idx], device=device)
            dst = torch.tensor(
                np.array([pos_dst.cpu().numpy()[idx]]),
                device=device,
            )

            n_id = torch.cat([src, dst]).unique()
            # print(f"Positive n_id: {n_id}")
            n_id, edge_index, e_id = neighbor_loader(n_id)
            # print(f"Positive edge_index: {edge_index}, Positive e_id: {e_id}")
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            # Get updated memory of all nodes involved in the computation.
            z, last_update = model["memory"](n_id)
            # print(f"z: {z.shape}, last_update: {last_update.shape}")
            # print(f"Real message shape: {data.msg[e_id].shape}")
            z = model["gnn"](
                z,
                last_update,
                edge_index,
                data.t[e_id].to(device),
                data.msg[e_id].to(device),
            )

            y_pred = model["link_pred"](z[assoc[src]], z[assoc[dst]])

            if src[0] == target_src and dst[0] == target_dst:
                predictions.append(y_pred[0, 0].item())
                timestamps.append(pos_t[idx].item())
                relevant_edge_indices.update(e_id.cpu().numpy().tolist())
            elif (
                src[0] == target_src
                or src[0] == target_dst
                or dst[0] == target_src
                or dst[0] == target_dst
            ):
                adjacent_event_timestamps.append(pos_t[idx].item())

            evaluate_negative(pos_t[idx])

        # Update memory and neighbor loader with ground-truth state.
        model["memory"].update_state(pos_src, pos_dst, pos_t, pos_msg)
        neighbor_loader.insert(pos_src, pos_dst)

    max_t = additional_negative_edges[-1][2] + 1
    evaluate_negative(max_t)

    print(len(predictions), len(predictions_neg))
    print(np.array(predictions_neg).shape)
    all_preds = np.array(predictions_neg)
    all_timestamps = np.array(timestamps_neg)
    # all_preds = np.concatenate((np.array(predictions), np.array(predictions_neg)))
    # all_timestamps = np.concatenate((np.array(timestamps), np.array(timestamps_neg)))
    sort = np.argsort(all_timestamps)

    all_preds = all_preds[sort]
    all_timestamps = all_timestamps[sort]
    print(f"Number of distinct predictions: {len(set(all_timestamps))}")
    atimes = np.array(timestamps)

    print(f"Length of all_preds: {len(all_preds)}")
    print(f"Number of additional negative edges: {len(additional_negative_edges)}")

    return [
        all_timestamps,
        all_preds,
        atimes,
        adjacent_event_timestamps,
        list(relevant_edge_indices),
    ]


# ==========
# ==========
# ==========


# Start...
start_overall = timeit.default_timer()

# ========== set parameters...
args, _ = get_args()
print("INFO: Arguments:", args)

DATA = args.data
LR = args.lr
BATCH_SIZE = args.bs
K_VALUE = args.k_value
NUM_EPOCH = args.num_epoch
SEED = args.seed
MEM_DIM = args.mem_dim
TIME_DIM = args.time_dim
EMB_DIM = args.emb_dim
TOLERANCE = args.tolerance
PATIENCE = args.patience
NUM_RUNS = args.num_run
NUM_NEIGHBORS = 10
TIME_ENCODER = args.time_encoder
MULTIPLIER = args.mul


MODEL_NAME = "TGN"

DO_REAL_TEST = True
# ==========

# set the device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# data loading
dataset = PyGLinkPropPredDataset(name=DATA, root="datasets")
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask
data = dataset.get_TemporalData()
data = data.to(device)
metric = dataset.eval_metric

train_data = data[train_mask]
val_data = data[val_mask]
test_data = data[test_mask]

test_pairs = list(zip(test_data.src.tolist(), test_data.dst.tolist()))
counts = {i: test_pairs.count(i) for i in set(test_pairs)}
biggest = sorted(set(test_pairs), key=lambda i: counts[i], reverse=True)

n_bins = 50

tgn_results = {}

run_all = False

for i in range(100, len(biggest), 200):
    target_src, target_dst = biggest[i]
    count = counts[(target_src, target_dst)]
    if count == 1:
        break
    print(
        f"INFO: Target source: {target_src}, Target destination: {target_dst}, Count: {count}"
    )
    additional_negative_edges = []

    time_range = test_data["t"].max() - test_data["t"].min()

    lower_bound = int(
        test_data["t"].min()
    )  # can't extend backwards without clashing with val/train
    upper_bound = int(test_data["t"].max() + time_range * 0.2)

    # step = (upper_bound - lower_bound) // n_bins
    step = 1
    print(f"Span of time: {test_data['t'].max() - test_data['t'].min()}")
    print(f"INFO: step: {step}")

    i = lower_bound
    while i <= upper_bound:
        additional_negative_edges.append((target_src, target_dst, i))
        i += step
    print(f"Number of additional negative edges: {len(additional_negative_edges)}")

    # Small batch size unimportant for train/val
    train_loader = TemporalDataLoader(train_data, batch_size=200)
    val_loader = TemporalDataLoader(val_data, batch_size=200)
    test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE)

    # Ensure to only sample actual destination nodes as negatives.
    min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

    # neighhorhood sampler
    neighbor_loader = LastNeighborLoader(
        data.num_nodes, size=NUM_NEIGHBORS, device=device
    )

    # define the model end-to-end
    memory = TGNMemory(
        data.num_nodes,
        data.msg.size(-1),
        MEM_DIM,
        TIME_DIM,
        message_module=IdentityMessage(data.msg.size(-1), MEM_DIM, TIME_DIM),
        aggregator_module=LastAggregator(),
        time_encoder=TIME_ENCODER,
        multiplier=MULTIPLIER,
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=MEM_DIM,
        out_channels=EMB_DIM,
        msg_dim=data.msg.size(-1),
        time_enc=memory.time_enc,
    ).to(device)

    link_pred = LinkPredictor(in_channels=EMB_DIM).to(device)

    model = {"memory": memory, "gnn": gnn, "link_pred": link_pred}
    print(model)

    # Helper vector to map global node indices to local ones.
    assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

    print("==========================================================")
    print(f"=================*** {MODEL_NAME}: LinkPropPred: {DATA} ***=============")
    print("==========================================================")

    evaluator = Evaluator(name=DATA)
    neg_sampler = dataset.negative_sampler

    # for saving the results...
    results_path = f"{osp.dirname(osp.abspath(__file__))}/saved_results"
    if not osp.exists(results_path):
        os.mkdir(results_path)
        print("INFO: Create directory {}".format(results_path))
    Path(results_path).mkdir(parents=True, exist_ok=True)
    results_filename = f"{results_path}/{MODEL_NAME}_{DATA}_results.json"

    for run_idx in range(NUM_RUNS):
        print(
            "-------------------------------------------------------------------------------"
        )
        print(f"INFO: >>>>> Run: {run_idx} <<<<<")
        start_run = timeit.default_timer()

        # set the seed for deterministic results...
        torch.manual_seed(run_idx + SEED)
        set_random_seed(run_idx + SEED)

        # define an early stopper
        save_model_dir = f"{osp.dirname(osp.abspath(__file__))}/saved_models/"
        save_model_id = (
            f"{MODEL_NAME}_{DATA}_{SEED}_{run_idx}_{TIME_ENCODER}_{MULTIPLIER}"
        )
        early_stopper = EarlyStopMonitor(
            save_model_dir=save_model_dir,
            save_model_id=save_model_id,
            tolerance=TOLERANCE,
            patience=PATIENCE,
        )

        # ==================================================== Test
        # first, load the best model
        early_stopper.load_checkpoint(model)

        model["memory"].reset_state()  # Start with a fresh memory.
        neighbor_loader.reset_state()  # Start with an empty graph.

        model["memory"].eval()
        model["gnn"].eval()
        model["link_pred"].eval()

        process_split(train_loader)
        process_split(val_loader)

        # loading the test negative samples
        dataset.load_test_ns()
        early_stopper.load_checkpoint(model)
        if DO_REAL_TEST:
            perf_metric_test = real_test(test_loader, neg_sampler, split_mode="test")
            print(f"Performance metric on test dataset: {perf_metric_test}")

        # final testing
        start_test = timeit.default_timer()
        prediction_results = test(
            test_loader,
            target_src,
            target_dst,
            additional_negative_edges,
        )

        if run_all:
            tgn_results[(target_src, target_dst)] = prediction_results
        else:
            print(f"Lenght of prediction results: {len(prediction_results[1])}")
            print(f"Number of events: {len(prediction_results[2])}")
            print(f"Number of adjacent events: {len(prediction_results[3])}")
            print(f"Number of relevant events: {len(prediction_results[4])}")

            one_hop_neighbor_timestamp_set = set(prediction_results[3])
            event_timestamp_set = set(prediction_results[2])

            hop0, hop1, hop2 = get_temporal_edge_times(
                dataset, target_src, target_dst, 2, mask=test_mask
            )

            for hop_threshold in range(4):
                totvar, totvar_per_sec = total_variation_per_unit_time(
                    [hop0, hop1, hop2][:hop_threshold],
                    prediction_results[1],
                    prediction_results[0],
                )

                print(f"TotalVar-{hop_threshold} = {totvar}")
                print(f"TotalVar/s-{hop_threshold} = {totvar_per_sec}")

            print(f"Length of hop0: {len(hop0)}")
            print(f"Length of hop1: {len(hop1)}")
            print(f"Length of hop2: {len(hop2)}")

            plt.plot(
                prediction_results[0],
                prediction_results[1],
                alpha=0.7,
                linewidth=1.5,
            )
            plt.xlabel("Time")
            plt.ylabel("Predicted link probability")
            plt.ylim(-0.01, 1.01)

            for etime in hop0:
                plt.axvline(x=etime, color="C1", ls="--", linewidth=1.0, alpha=1.0)

            for etime in hop1:
                plt.axvline(x=etime, color="C2", ls="--", linewidth=1.0, alpha=1.0)

            for etime in hop2:
                plt.axvline(x=etime, color="C3", ls="--", linewidth=1.0, alpha=1.0)

            plt.show()

        print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
        test_time = timeit.default_timer() - start_test
        print(f"\tTest: Elapsed Time (s): {test_time: .4f}")

    print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")
    print("==============================================================")

np.save(f"tgn_results_{SEED}.npy", tgn_results)
