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
import os.path as osp
import numpy as np

import torch
from torch_geometric.loader import TemporalDataLoader

# internal imports
from tgb.utils.utils import get_args, set_random_seed, save_results
from tgb.linkproppred.evaluate import Evaluator
from modules.decoder import LinkPredictor
from modules.emb_module import GraphAttentionEmbedding
from modules.msg_func import IdentityMessage
from modules.msg_agg import LastAggregator
from modules.neighbor_loader import LastNeighborLoader
from modules.memory_module import TGNMemory, DyRepMemory
from modules.early_stopping import EarlyStopMonitor
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path

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
    src_dst_pairs,
    negative_edge_times,
):
    src_dst_pairs_tensor = torch.tensor(src_dst_pairs, device=device)

    def evaluate_negative(end_time):
        src_n = src_dst_pairs_tensor[:, 0]
        dst_n = src_dst_pairs_tensor[:, 1]

        t_neg_ls = []
        while len(negative_edge_times) > 0 and negative_edge_times[0] <= end_time:
            t_neg = negative_edge_times.pop(0)
            t_neg_ls.append(t_neg)  # all t_negs are the same

        n_id = torch.cat(
            [src_n, dst_n],
        ).unique()
        if len(t_neg_ls) != 0:
            n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            # Get updated memory of all nodes involved in the computation.
            s, last_update = model["memory"](n_id)

            y_pred_negs = []
            for tn in t_neg_ls:
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

                y_pred_neg = model["link_pred"](z[assoc[src_n]], z[assoc[dst_n]])
                y_pred_neg = y_pred_neg.squeeze(-1).unsqueeze(0)
                y_pred_negs.append(y_pred_neg.cpu())

            y_pred_neg = torch.cat(y_pred_negs)
            predictions_neg.extend(y_pred_neg.squeeze(-1).cpu().numpy())
            timestamps_neg.extend(np.array(t_neg_ls))

    model["memory"].eval()
    model["gnn"].eval()
    model["link_pred"].eval()

    predictions_neg = []
    timestamps_neg = []

    for pos_batch in tqdm(loader):
        pos_src, pos_dst, pos_t, pos_msg = (
            pos_batch.src,
            pos_batch.dst,
            pos_batch.t,
            pos_batch.msg,
        )

        for idx, _ in enumerate(pos_batch):
            evaluate_negative(pos_t[idx])

        # Update memory and neighbor loader with ground-truth state.
        model["memory"].update_state(pos_src, pos_dst, pos_t, pos_msg)
        neighbor_loader.insert(pos_src, pos_dst)

    # max_t = max(negative_edge_times) + 1
    # evaluate_negative(max_t)

    return predictions_neg, timestamps_neg


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

TIME_STEP = args.time_step
EDGE_STEP = args.edge_step

MODEL_NAME = args.model

DO_REAL_TEST = False
EXCLUDE_EVENTS = False

metric_output_dir = Path(f"{osp.dirname(osp.abspath(__file__))}/results/")
metric_output_dir.mkdir(parents=True, exist_ok=True)
metric_output_path = (
    metric_output_dir
    / f"{MODEL_NAME}_{DATA}_{SEED}_{TIME_ENCODER}_{MULTIPLIER}_{TIME_STEP}_{EDGE_STEP}.txt"
)

f = open(str(metric_output_path), "w")


step_difference_across_runs = []
total_variation_across_runs = []
total_variation_across_runs_with_hop = [[], [], []]
for run_idx in range(NUM_RUNS):
    # ==========
    # set the seed for deterministic results...
    torch.manual_seed(run_idx + SEED)
    set_random_seed(run_idx + SEED)

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

    print(f"Number of unqiue edges: {len(biggest)}")
    f.write(f"Number of unqiue edges: {len(biggest)}\n")

    src_dst_pairs = []
    for i in range(0, len(biggest), EDGE_STEP):
        src, dst = biggest[i]
        src_dst_pairs.append([src, dst])
    print(f"Number of selected edges: {len(src_dst_pairs)}")
    f.write(f"Number of selected edges: {len(src_dst_pairs)}\n")

    time_range = test_data["t"].max() - test_data["t"].min()
    lower_bound = int(
        test_data["t"].min()
    )  # can't extend backwards without clashing with val/train
    upper_bound = int(test_data["t"].max())

    step = TIME_STEP
    print(f"Span of time: {upper_bound - lower_bound}")
    f.write(f"Span of time: {upper_bound - lower_bound}\n")
    print(f"INFO: step: {step}")
    f.write(f"INFO: step: {step}\n")

    negative_edge_times = list(range(lower_bound, upper_bound + 1, step))
    print(
        f"INFO: Number of negative timestamps to do inference at: {len(negative_edge_times)}"
    )
    f.write(
        f"INFO: Number of negative timestamps to do inference at: {len(negative_edge_times)}\n"
    )

    # Small batch size unimportant for train/val
    train_loader = TemporalDataLoader(train_data, batch_size=200)
    val_loader = TemporalDataLoader(val_data, batch_size=200)
    test_loader = TemporalDataLoader(test_data, batch_size=1)

    # Ensure to only sample actual destination nodes as negatives.
    min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

    # neighhorhood sampler
    neighbor_loader = LastNeighborLoader(
        data.num_nodes, size=NUM_NEIGHBORS, device=device
    )

    # define the model end-to-end
    if MODEL_NAME == "TGN":
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
    elif MODEL_NAME == "DyRep":
        USE_SRC_EMB_IN_MSG = False
        USE_DST_EMB_IN_MSG = True
        memory = DyRepMemory(
            data.num_nodes,
            data.msg.size(-1),
            MEM_DIM,
            TIME_DIM,
            message_module=IdentityMessage(data.msg.size(-1), MEM_DIM, TIME_DIM),
            aggregator_module=LastAggregator(),
            memory_updater_type="rnn",
            use_src_emb_in_msg=USE_SRC_EMB_IN_MSG,
            use_dst_emb_in_msg=USE_DST_EMB_IN_MSG,
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
    f.write(f"{model}\n")

    # Helper vector to map global node indices to local ones.
    assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

    print("==========================================================")
    print(f"=================*** {MODEL_NAME}: LinkPropPred: {DATA} ***=============")
    f.write(
        f"=================*** {MODEL_NAME}: LinkPropPred: {DATA} ***=============\n"
    )
    print("==========================================================")

    evaluator = Evaluator(name=DATA)
    neg_sampler = dataset.negative_sampler

    print(
        "-------------------------------------------------------------------------------"
    )
    print(f"INFO: >>>>> Run: {run_idx} <<<<<")
    f.write(f"INFO: >>>>> Run: {run_idx} <<<<<\n")
    start_run = timeit.default_timer()

    # define an early stopper
    save_model_dir = f"{osp.dirname(osp.abspath(__file__))}/saved_models/"
    save_model_id = f"{MODEL_NAME}_{DATA}_{SEED}_{run_idx}_{TIME_ENCODER}_{MULTIPLIER}"
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

    predictions_neg, timestamps_neg = test(
        test_loader, src_dst_pairs, negative_edge_times
    )
    predictions_neg = np.array(predictions_neg)
    timestamps_neg = np.array(timestamps_neg)
    print(predictions_neg.shape)

    if not EXCLUDE_EVENTS:

        def total_variation_batch(predictions_batch):
            return np.sum(
                np.abs(predictions_batch[1:, :] - predictions_batch[:-1, :]), axis=0
            ) / (upper_bound - lower_bound)

        mean_total_variations = total_variation_batch(predictions_neg)
        print(mean_total_variations.shape)
        mean_metric_over_all_unique_edges = np.mean(mean_total_variations)
        print(
            f"Mean total variation over all edges: {mean_metric_over_all_unique_edges}"
        )
        f.write(
            f"Mean total variation over all edges: {mean_metric_over_all_unique_edges}\n"
        )
    else:
        mean_step_differences = []
        mean_total_variations = []

        mean_total_variations_with_hop = [[], [], []]

        def total_variation_single(predictions):
            return np.mean(np.abs(predictions[1:] - predictions[:-1]))

        for i in tqdm(range(len(src_dst_pairs))):
            src, dst = src_dst_pairs[i]
            predictions = predictions_neg[:, i]

            total_variation = total_variation_per_unit_time(
                [], np.array(predictions), timestamps_neg
            )[1]
            # print(f"Total variation for edge {src}-{dst}: {total_variation}")
            f.write(f"Total variation for edge {src}-{dst}: {total_variation}\n")
            mean_total_variations.append(total_variation)

            hop0, hop1, hop2 = get_temporal_edge_times(
                dataset, src, dst, 2, mask=test_mask
            )

            for hop_threshold in range(1, 4):
                totvar, totvar_per_sec = total_variation_per_unit_time(
                    [hop0, hop1, hop2][:hop_threshold],
                    predictions,
                    timestamps_neg,
                )
                mean_total_variations_with_hop[hop_threshold - 1].append(totvar_per_sec)

                # f.write(f"TotalVar-{hop_threshold} = {totvar}")
                f.write(f"TotalVar/s-{hop_threshold} = {totvar_per_sec}")

        mean_total_variations = np.array(mean_total_variations)
        mean_metric_over_all_unique_edges = np.mean(mean_total_variations)
        print(
            f"Mean total variation over all edges: {mean_metric_over_all_unique_edges}"
        )
        f.write(
            f"Mean total variation over all edges: {mean_metric_over_all_unique_edges}\n"
        )

    # step_difference_across_runs.append(mean_metric_over_all_unique_edges)
    total_variation_across_runs.append(mean_metric_over_all_unique_edges)

    if EXCLUDE_EVENTS:
        for hop in range(1, 4):
            mean_metric_over_all_unique_edges_with_hop = np.mean(
                np.array(mean_total_variations_with_hop[hop - 1])
            )
            print(
                f"Mean total variation over all edges excluding hop {hop}: {mean_metric_over_all_unique_edges_with_hop}"
            )
            f.write(
                f"Mean total variation over all edges excluding hop {hop}: {mean_metric_over_all_unique_edges_with_hop}\n"
            )
            total_variation_across_runs_with_hop[hop - 1].append(
                mean_metric_over_all_unique_edges_with_hop
            )

    print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")
    f.write(
        f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}\n"
    )
    print("==============================================================")


total_variation_across_runs = np.array(total_variation_across_runs)
total_variation_mean = np.mean(total_variation_across_runs)
total_variation_std = np.std(total_variation_across_runs)
print(f"total variation mean across runs: {total_variation_mean}")
f.write(f"total variation mean across runs: {total_variation_mean}\n")
print(f"total variation std across runs: {total_variation_std}")
f.write(f"total variation std across runs: {total_variation_std}\n")

if EXCLUDE_EVENTS:
    for hop in range(1, 4):
        total_variation_across_runs_with_single_hop = np.array(
            total_variation_across_runs_with_hop[hop - 1]
        )
        total_variation_mean = np.mean(total_variation_across_runs_with_single_hop)
        total_variation_std = np.std(total_variation_across_runs_with_single_hop)
        print(
            f"total variation mean across runs (excluding hop {hop}): {total_variation_mean}"
        )
        f.write(
            f"total variation mean across runs (excluding hop {hop}): {total_variation_mean}\n"
        )
        print(
            f"total variation std across runs (excluding hop {hop}): {total_variation_std}"
        )
        f.write(
            f"total variation std across runs (excluding hop {hop}): {total_variation_std}\n"
        )

f.close()
