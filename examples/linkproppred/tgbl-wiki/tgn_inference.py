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


# ==========
# ========== Define helper function...
# ==========

target_src = 1206
target_dst = 8734


@torch.no_grad()
def test(loader, neg_sampler, positive_sample_mask, split_mode):
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

    predictions = []
    timestamps = []

    avg_prob = 0.0
    cnt = 0

    target_edge_cnt = 0

    num_edge_for_memory_update = 0

    for pos_batch in loader:
        pos_src, pos_dst, pos_t, pos_msg = (
            pos_batch.src,
            pos_batch.dst,
            pos_batch.t,
            pos_batch.msg,
        )

        # neg_batch_list = neg_sampler.query_batch(
        #     pos_src, pos_dst, pos_t, split_mode=split_mode
        # )

        negative_indices = set()

        for idx, _ in enumerate(pos_batch):
            src = torch.full((1,), pos_src[idx], device=device)
            dst = torch.tensor(
                np.array([pos_dst.cpu().numpy()[idx]]),
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

            avg_prob += y_pred[0][0].cpu().numpy()
            cnt += 1
            if src[0] == target_src and dst[0] == target_dst:
                predictions.append(y_pred[0][0].item())
                timestamps.append(pos_t[idx].item())
                if (
                    positive_sample_mask is not None
                    and not positive_sample_mask[target_edge_cnt]
                ):
                    negative_indices.add(idx)
                    avg_prob -= y_pred[0][0].cpu().numpy()
                    cnt -= 1
                target_edge_cnt += 1

            # compute MRR
            input_dict = {
                "y_pred_pos": np.array([y_pred[0, :].squeeze(dim=-1).cpu()]),
                "y_pred_neg": np.array(y_pred[1:, :].squeeze(dim=-1).cpu()),
                "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])

        # Update memory and neighbor loader with ground-truth state.

        if positive_sample_mask is not None:
            true_pos_src = []
            true_pos_dst = []
            true_pos_t = []
            true_pos_msg = []

            for i in range(len(pos_src)):
                if i in negative_indices:
                    continue
                true_pos_src.append(pos_src[i])
                true_pos_dst.append(pos_dst[i])
                true_pos_t.append(pos_t[i])
                true_pos_msg.append(pos_msg[i])

            pos_src = torch.tensor(np.array(true_pos_src))
            pos_dst = torch.tensor(np.array(true_pos_dst))
            pos_t = torch.tensor(np.array(true_pos_t))
            pos_msg = torch.tensor(np.array(true_pos_msg))

            # print(f"Length of edges used to update memory: {pos_msg.shape}")
            num_edge_for_memory_update += pos_msg.shape[0]

        if len(pos_src) == 0:
            continue
        model["memory"].update_state(pos_src, pos_dst, pos_t, pos_msg)
        neighbor_loader.insert(pos_src, pos_dst)

    perf_metrics = float(torch.tensor(perf_list).mean())

    print(f"timestamps: {timestamps}")
    print(f"predictions: {predictions}")
    print(f"avg_prob: {avg_prob / cnt}")
    print(f"num_edge_for_memory_update: {num_edge_for_memory_update}")
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

print(f"Length of test data: {len(test_data['src'])}")

add_negative_samples = False
if add_negative_samples:
    n_bins = 50
    step = (test_data["t"].max() - test_data["t"].min()) // n_bins
    print(f"Span of time: {test_data['t'].max() - test_data['t'].min()}")
    print(f"INFO: step: {step}")

    positive_sample_mask = []
    new_src = []
    new_dst = []
    new_t = []
    new_msg = []

    index = 0
    added_cnt = 0
    for i in range(test_data["t"].min(), test_data["t"].max(), step):
        while not i < test_data["t"][index + 1]:
            if (
                test_data["src"][index].item() == target_src
                and test_data["dst"][index].item() == target_dst
            ):
                positive_sample_mask.append(True)
            new_src.append(test_data["src"][index].item())
            new_dst.append(test_data["dst"][index].item())
            new_t.append(test_data["t"][index].item())
            # print(test_data["msg"][index])
            new_msg.append(test_data["msg"][index])
            index += 1

        new_src.append(test_data["src"][index].item())
        new_dst.append(test_data["dst"][index].item())
        new_t.append(test_data["t"][index].item())
        new_msg.append(test_data["msg"][index])

        positive_sample_mask.append(False)
        new_src.append(target_src)
        new_dst.append(target_dst)
        new_t.append(i)
        new_msg.append(torch.zeros(172, device=device))
        added_cnt += 1
        index += 1

    while not index == len(test_data["t"]):
        new_src.append(test_data["src"][index].item())
        new_dst.append(test_data["dst"][index].item())
        new_t.append(test_data["t"][index].item())
        new_msg.append(torch.zeros(172, device=device))
        index += 1

    test_data["src"] = torch.tensor(new_src)
    test_data["dst"] = torch.tensor(new_dst)
    test_data["t"] = torch.tensor(new_t)
    test_data["msg"] = torch.stack(new_msg)

    print(test_data["src"].shape)
    print(test_data["msg"].shape)

    print(f"added_cnt: {added_cnt}")
    print(f"Length of test data: {len(test_data['src'])}")

    positive_sample_mask = torch.tensor(positive_sample_mask)
    print(f"positive_sample_mask: {positive_sample_mask}")

    assert (
        len(test_data["src"])
        == len(test_data["dst"])
        == len(test_data["t"])
        == len(test_data["msg"])
    )

train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)
val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE)

# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

# neighhorhood sampler
neighbor_loader = LastNeighborLoader(data.num_nodes, size=NUM_NEIGHBORS, device=device)

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
    if add_negative_samples:
        perf_metric_test = test(
            test_loader, neg_sampler, positive_sample_mask, split_mode="test"
        )
    else:
        perf_metric_test = test(test_loader, neg_sampler, None, split_mode="test")

    print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
    print(f"\tTest: {metric}: {perf_metric_test: .4f}")
    test_time = timeit.default_timer() - start_test
    print(f"\tTest: Elapsed Time (s): {test_time: .4f}")

print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")
print("==============================================================")
