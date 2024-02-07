"""
Dynamic Link Prediction with a TGN model with Early Stopping
Reference: 
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py

command for an example run:
    python examples/linkproppred/tgbl-wiki/tgn.py --data "tgbl-wiki" --num_run 1 --seed 1
"""

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

# ==========
# ========== Define helper function...
# ==========


def train():
    r"""
    Training procedure for TGN model
    This function uses some objects that are globally defined in the current scrips

    Parameters:
        None
    Returns:
        None

    """

    model["memory"].train()
    model["gnn"].train()
    model["link_pred"].train()

    model["memory"].reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        # Sample negative destination nodes.
        neg_dst = torch.randint(
            min_dst_idx,
            max_dst_idx + 1,
            (src.size(0),),
            dtype=torch.long,
            device=device,
        )

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
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

        pos_out = model["link_pred"](z[assoc[src]], z[assoc[pos_dst]])
        neg_out = model["link_pred"](z[assoc[src]], z[assoc[neg_dst]])

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        model["memory"].update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        model["memory"].detach()
        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events


@torch.no_grad()
def test(
    loader,
    target_src,
    target_dst,
    additional_negative_edges,
    neg_sampler,
    tgn_results,
    split_mode,
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
    model["memory"].eval()
    model["gnn"].eval()
    model["link_pred"].eval()

    predictions = []
    timestamps = []

    predictions_neg = []
    timestamps_neg = []

    perf_list = []

    for pos_batch in loader:
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

            while (
                len(additional_negative_edges) > 0
                and additional_negative_edges[0][2] < pos_t[idx]
            ):
                src_n, dst_n, t_neg = additional_negative_edges.pop(0)
                t_neg = torch.tensor(t_neg, device=device)

                src_n = torch.full((1,), src_n, device=device)
                dst_n = torch.full((1,), dst_n, device=device)

                n_id = torch.cat([src_n, dst_n]).unique()
                # print(f"n_id: {n_id}")
                n_id, edge_index, e_id = neighbor_loader(n_id)
                # print(
                #     f"n_id: {n_id}, edge_index: {edge_index}, e_id: {e_id}, t_neg: {t_neg}, pos_t: {pos_t[idx]}"
                # )
                assoc[n_id] = torch.arange(n_id.size(0), device=device)

                # Get updated memory of all nodes involved in the computation.
                z, last_update = model["memory"](n_id)
                # print(f"z: {z.shape}, last_update: {last_update.shape}")
                z = model["gnn"](
                    z,
                    last_update,
                    edge_index,
                    data.t[e_id].to(device),
                    data.msg[e_id].to(device),
                )

                y_pred_neg = model["link_pred"](z[assoc[src_n]], z[assoc[dst_n]])
                # print(f"Negative prediction: {y_pred_neg}")

                predictions_neg.append(y_pred_neg.item())
                timestamps_neg.append(t_neg.item())

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

    # print("Positive predictions: ", predictions)
    # print("Positive timestamps: ", timestamps)
    # print("Negative predictions: ", predictions_neg)
    # print("Negative timestamps: ", timestamps_neg)

    all_preds = np.concatenate((np.array(predictions), np.array(predictions_neg)))
    all_timestamps = np.concatenate((np.array(timestamps), np.array(timestamps_neg)))
    sort = np.argsort(all_timestamps)

    all_preds = all_preds[sort]
    all_timestamps = all_timestamps[sort]
    atimes = np.array(timestamps)

    tgn_results[(target_src, target_dst)] = [all_timestamps, all_preds, atimes]

    return perf_metrics


# ==========
# ==========
# ==========


# Start...
start_overall = timeit.default_timer()

# ========== set parameters...
args, _ = get_args()
print("INFO: Arguments:", args)

DATA = "tgbl-wiki"
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


MODEL_NAME = "TGN"
# ==========

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

for i in range(0, len(biggest), 50):
    target_src, target_dst = biggest[i]
    count = counts[(target_src, target_dst)]
    if count == 1:
        break

    print(
        f"INFO: Target source: {target_src}, Target destination: {target_dst}, Count: {count}"
    )

    additional_negative_edges = []

    time_range = test_data["t"].max() - test_data["t"].min()
    lower_bound = test_data["t"].min() - time_range * 0.2
    upper_bound = test_data["t"].max() + time_range * 0.2

    step = (upper_bound - lower_bound) // n_bins
    print(f"Span of time: {test_data['t'].max() - test_data['t'].min()}")
    print(f"INFO: step: {step}")
    for i in range(n_bins):
        additional_negative_edges.append(
            (target_src, target_dst, lower_bound + i * step)
        )
    # print(f"Length of additional negative edges: {len(additional_negative_edges)}")
    # print(f"First element of additional negative edges: {additional_negative_edges[0]}")
    # print(
    #     f"Second element of additional negative edges: {additional_negative_edges[1]}"
    # )

    train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)
    val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
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
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=MEM_DIM,
        out_channels=EMB_DIM,
        msg_dim=data.msg.size(-1),
        time_enc=memory.time_enc,
    ).to(device)

    link_pred = LinkPredictor(in_channels=EMB_DIM).to(device)

    model = {"memory": memory, "gnn": gnn, "link_pred": link_pred}

    optimizer = torch.optim.Adam(
        set(model["memory"].parameters())
        | set(model["gnn"].parameters())
        | set(model["link_pred"].parameters()),
        lr=LR,
    )
    criterion = torch.nn.BCEWithLogitsLoss()

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
        save_model_id = f"{MODEL_NAME}_{DATA}_{SEED}_{run_idx}"
        early_stopper = EarlyStopMonitor(
            save_model_dir=save_model_dir,
            save_model_id=save_model_id,
            tolerance=TOLERANCE,
            patience=PATIENCE,
        )

        # ==================================================== Test
        # first, load the best model
        early_stopper.load_checkpoint(model)

        # loading the test negative samples
        dataset.load_test_ns()

        # final testing
        start_test = timeit.default_timer()
        perf_metric_test = test(
            test_loader,
            target_src,
            target_dst,
            additional_negative_edges,
            neg_sampler,
            tgn_results,
            split_mode="test",
        )

        print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
        print(f"\tTest: {metric}: {perf_metric_test: .4f}")
        test_time = timeit.default_timer() - start_test
        print(f"\tTest: Elapsed Time (s): {test_time: .4f}")

    print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")
    print("==============================================================")

np.save(f"tgn_results_{SEED}.npy", tgn_results)
