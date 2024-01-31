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


@torch.no_grad()
def test(loader, neg_sampler, split_mode):
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

    for pos_batch in loader:
        # pos_src, pos_dst, pos_t, pos_msg = (
        #     pos_batch.src,
        #     pos_batch.dst,
        #     pos_batch.t,
        #     pos_batch.msg,
        # )
        pos_src = torch.tensor([0, 1], dtype=torch.long, device=device)
        pos_dst = torch.tensor([8227, 8228], dtype=torch.long, device=device)
        pos_t = torch.tensor([0.0, 36.0], device=device)
        pos_msg = torch.tensor(
            [
                [
                    -0.17506251,
                    -0.17667764,
                    -0.93709077,
                    -0.38192566,
                    0.0,
                    -0.63653506,
                    1.05239649,
                    -0.16937994,
                    -0.19303961,
                    -0.16923411,
                    -0.82894224,
                    -0.17509305,
                    -0.23967511,
                    -0.08175889,
                    -0.43849158,
                    -0.26522708,
                    -0.2705335,
                    -0.76624089,
                    -0.2308179,
                    -0.20839674,
                    -0.03145804,
                    -0.1460524,
                    -0.12591898,
                    -0.11499875,
                    -0.07366619,
                    -0.080377,
                    -0.03997727,
                    -0.08641651,
                    -0.07646035,
                    -0.11533572,
                    -0.11297558,
                    -0.12955636,
                    -0.114021,
                    -0.12691053,
                    -0.11293875,
                    -0.0863202,
                    -0.12647963,
                    -0.14023702,
                    -0.12746241,
                    -0.10123371,
                    -0.12888967,
                    -0.12375016,
                    -0.03667754,
                    -0.13216314,
                    -0.09182764,
                    -0.07686611,
                    -0.08342218,
                    -0.1326532,
                    -0.13227117,
                    -0.11352337,
                    -0.0828937,
                    -0.10053204,
                    -0.07578398,
                    -0.14363464,
                    -0.12752756,
                    -0.1284549,
                    -0.09904575,
                    -0.11245651,
                    -0.11828353,
                    -0.10532186,
                    -0.11722814,
                    -0.1053035,
                    -0.13007721,
                    -0.11733481,
                    -0.10045207,
                    -0.08298721,
                    -0.10977516,
                    -0.08251948,
                    -0.09015049,
                    -0.06180688,
                    -0.07659462,
                    -0.15440477,
                    -0.13370903,
                    -0.14931784,
                    -0.14752621,
                    -0.15763136,
                    -0.13390825,
                    -0.14212796,
                    -0.10124252,
                    -0.09423485,
                    -0.08458487,
                    -0.08920547,
                    -0.05786022,
                    -0.0678592,
                    -0.05542865,
                    -0.05232922,
                    -0.08558759,
                    -0.08400994,
                    1.75645384,
                    -0.28993879,
                    0.0,
                    -0.41798993,
                    -1.41094806,
                    -0.10921465,
                    -0.12200883,
                    -0.04039036,
                    1.82576237,
                    -0.1209567,
                    -0.17511816,
                    -0.06204122,
                    -0.26592222,
                    -0.10833547,
                    -0.11851638,
                    1.19621267,
                    -0.15192837,
                    -0.1434505,
                    -0.00926212,
                    -0.10337834,
                    -0.08909862,
                    -0.08199795,
                    -0.0526049,
                    -0.05569965,
                    -0.03084629,
                    -0.06095729,
                    -0.0538461,
                    -0.08162444,
                    -0.07844069,
                    -0.08799932,
                    -0.07942915,
                    -0.08881298,
                    -0.07198582,
                    -0.06538741,
                    -0.08999549,
                    -0.09740804,
                    -0.09306396,
                    -0.07634347,
                    -0.08665537,
                    -0.08535061,
                    -0.03641195,
                    -0.08736506,
                    -0.05944714,
                    -0.05338855,
                    -0.05568519,
                    -0.08714054,
                    -0.08925076,
                    -0.07339566,
                    -0.05874054,
                    -0.06798958,
                    -0.03871806,
                    -0.0974486,
                    -0.08757796,
                    -0.09096852,
                    -0.07200306,
                    -0.08123793,
                    -0.08174704,
                    -0.06876709,
                    -0.08757911,
                    -0.07566231,
                    -0.08267461,
                    -0.07205032,
                    -0.06846234,
                    -0.04557322,
                    -0.07326132,
                    -0.05312438,
                    -0.06232004,
                    -0.0498802,
                    -0.05060037,
                    -0.10539429,
                    -0.09039525,
                    -0.09981604,
                    -0.10288771,
                    -0.10290329,
                    -0.09011459,
                    -0.09606808,
                    -0.06813565,
                    -0.06083513,
                    -0.05839396,
                    -0.06207875,
                    -0.0446735,
                    -0.05046383,
                    -0.04144796,
                    -0.03877482,
                ],
                [
                    -0.17506251,
                    -0.17667764,
                    -0.93709077,
                    -0.38192566,
                    0.0,
                    -0.63653506,
                    1.05239649,
                    -0.16937994,
                    -0.19303961,
                    -0.16923411,
                    -0.82894224,
                    -0.17509305,
                    -0.23967511,
                    -0.08175889,
                    -0.43849158,
                    -0.26522708,
                    -0.2705335,
                    -0.76624089,
                    -0.2308179,
                    -0.20839674,
                    -0.03145804,
                    -0.1460524,
                    -0.12591898,
                    -0.11499875,
                    -0.07366619,
                    -0.080377,
                    -0.03997727,
                    -0.08641651,
                    -0.07646035,
                    -0.11533572,
                    -0.11297558,
                    -0.12955636,
                    -0.114021,
                    -0.12691053,
                    -0.11293875,
                    -0.0863202,
                    -0.12647963,
                    -0.14023702,
                    -0.12746241,
                    -0.10123371,
                    -0.12888967,
                    -0.12375016,
                    -0.03667754,
                    -0.13216314,
                    -0.09182764,
                    -0.07686611,
                    -0.08342218,
                    -0.1326532,
                    -0.13227117,
                    -0.11352337,
                    -0.0828937,
                    -0.10053204,
                    -0.07578398,
                    -0.14363464,
                    -0.12752756,
                    -0.1284549,
                    -0.09904575,
                    -0.11245651,
                    -0.11828353,
                    -0.10532186,
                    -0.11722814,
                    -0.1053035,
                    -0.13007721,
                    -0.11733481,
                    -0.10045207,
                    -0.08298721,
                    -0.10977516,
                    -0.08251948,
                    -0.09015049,
                    -0.06180688,
                    -0.07659462,
                    -0.15440477,
                    -0.13370903,
                    -0.14931784,
                    -0.14752621,
                    -0.15763136,
                    -0.13390825,
                    -0.14212796,
                    -0.10124252,
                    -0.09423485,
                    -0.08458487,
                    -0.08920547,
                    -0.05786022,
                    -0.0678592,
                    -0.05542865,
                    -0.05232922,
                    -0.11414682,
                    -0.11528617,
                    -0.64961485,
                    -0.28993879,
                    0.0,
                    -0.41798993,
                    0.72326147,
                    -0.10921465,
                    -0.12200883,
                    -0.10923364,
                    -0.5597856,
                    -0.1209567,
                    -0.17511816,
                    -0.06204122,
                    -0.26592222,
                    -0.15539393,
                    -0.15704401,
                    -0.51589792,
                    -0.15192837,
                    -0.1434505,
                    -0.00926212,
                    -0.10337834,
                    -0.08909862,
                    -0.08199795,
                    -0.0526049,
                    -0.05569965,
                    -0.03084629,
                    -0.06095729,
                    -0.0538461,
                    -0.08162444,
                    -0.07844069,
                    -0.08799932,
                    -0.07942915,
                    -0.08881298,
                    -0.07198582,
                    -0.06538741,
                    -0.08999549,
                    -0.09740804,
                    -0.09306396,
                    -0.07634347,
                    -0.08665537,
                    -0.08535061,
                    -0.03641195,
                    -0.08736506,
                    -0.05944714,
                    -0.05338855,
                    -0.05568519,
                    -0.08714054,
                    -0.08925076,
                    -0.07339566,
                    -0.05874054,
                    -0.06798958,
                    -0.03871806,
                    -0.0974486,
                    -0.08757796,
                    -0.09096852,
                    -0.07200306,
                    -0.08123793,
                    -0.08174704,
                    -0.06876709,
                    -0.08757911,
                    -0.07566231,
                    -0.08267461,
                    -0.07205032,
                    -0.06846234,
                    -0.04557322,
                    -0.07326132,
                    -0.05312438,
                    -0.06232004,
                    -0.0498802,
                    -0.05060037,
                    -0.10539429,
                    -0.09039525,
                    -0.09981604,
                    -0.10288771,
                    -0.10290329,
                    -0.09011459,
                    -0.09606808,
                    -0.06813565,
                    -0.06083513,
                    -0.05839396,
                    -0.06207875,
                    -0.0446735,
                    -0.05046383,
                    -0.04144796,
                    -0.03877482,
                ],
            ],
            device=device,
        )

        print(
            f"pos_src: {pos_src.shape}, pos_dst: {pos_dst.shape}, pos_t: {pos_t.shape}, pos_msg: {pos_msg.shape}"
        )

        # neg_batch_list = neg_sampler.query_batch(
        #     pos_src, pos_dst, pos_t, split_mode=split_mode
        # )

        # for idx, neg_batch in enumerate(neg_batch_list):
        neg_batch = np.array([])
        idx = np.array([0])
        src = pos_src
        dst = pos_dst

        # src = torch.full((1,), pos_src[idx], device=device)
        # dst = torch.tensor(
        #     np.concatenate(
        #         ([np.array([pos_dst.cpu().numpy()[idx]]), np.array(neg_batch)]),
        #         axis=0,
        #     ),
        #     device=device,
        # )

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
        print(f"y_pred: {y_pred.shape}, {y_pred}")

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
        break

    perf_metrics = float(torch.tensor(perf_list).mean())

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
    perf_metric_test = test(test_loader, neg_sampler, split_mode="test")

    print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
    print(f"\tTest: {metric}: {perf_metric_test: .4f}")
    test_time = timeit.default_timer() - start_test
    print(f"\tTest: Elapsed Time (s): {test_time: .4f}")

print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")
print("==============================================================")
