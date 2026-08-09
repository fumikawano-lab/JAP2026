import os
import csv
import json

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import time
from torch_scatter import scatter_add
from torch_geometric.nn import GATConv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import random
from itertools import combinations
from scipy.spatial.distance import cosine
from collections import defaultdict, deque
from collections import Counter
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed
from math import comb
from tqdm import tqdm
from deap import creator, base, tools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_node_features = torch.load("./data/pt_data/model_features.pt")

model_target_edge_index = torch.load("./data/pt_data/edge_index.pt").to(device)
model_target_edge_attr = torch.load("./data/pt_data/edge_attr.pt").to(device).float()  

target_node_features = torch.load("./data/pt_data/target_features.pt")

target_edge_index_cpu = torch.load("./data/pt_data/target_edge_index.pt").long()
target_edge_index = target_edge_index_cpu.to(device)

target_edge_attr = torch.load("./data/pt_data/target_edge_attr.pt").float()

target_src_unique, target_src_counts = torch.unique(
    target_edge_index_cpu[0],
    return_counts=True
)

TARGET_OUT_DEGREE_DICT = {
    int(node_id): int(count)
    for node_id, count in zip(target_src_unique.tolist(), target_src_counts.tolist())
}

print(f"[LOG] TARGET_OUT_DEGREE_DICT constructed: {len(TARGET_OUT_DEGREE_DICT)} target src nodes")


def get_model_features(edge_index, model_features):
    source_nodes = edge_index[0].tolist() 
    extracted_features = []

    for model_id in source_nodes:
        if model_id in model_features:
            model_features_tensor = model_features[model_id].to(device).float() 
        else:
            model_features_tensor = torch.zeros((768,), dtype=torch.float32, device=device)  
        extracted_features.append(model_features_tensor)

    if len(extracted_features) == 0:
        return torch.empty((0, 768), dtype=torch.float32, device=device)

    model_feature_tensor = torch.stack(extracted_features).to(device).float() 
    return model_feature_tensor

def get_target_features(edge_index, target_node_features):
    target_nodes = edge_index[1].tolist()  
    extracted_features = []

    for target_id in target_nodes:
        if target_id in target_node_features:
            target_features_tensor = target_node_features[target_id].to(device).float()  
        else:
            target_features_tensor = torch.zeros((768,), dtype=torch.float32, device=device) 
        extracted_features.append(target_features_tensor)

    if len(extracted_features) == 0:
        return torch.empty((0, 768), dtype=torch.float32, device=device)

    target_feature_tensor = torch.stack(extracted_features).to(device).float() 
    return target_feature_tensor

loss_records = []

edge_weight_list = []
edge_weight_index_list = []

model_node_features = get_model_features(model_target_edge_index, model_node_features)

target_node_features = get_target_features(model_target_edge_index, target_node_features)

model_in_channels = model_node_features.shape[1]
target_in_channels = target_node_features.shape[1]
edge_in_channels = model_target_edge_attr.shape[1]
target_edge_in_channels = target_edge_attr.shape[1]

class ModelTargetAttentionGAT(nn.Module):
    def __init__(self, model_in_channels, target_in_channels, edge_in_channels, hidden_channels=256, heads=8, total_epochs=10):
        super(ModelTargetAttentionGAT, self).__init__()

        self.hidden_channels = hidden_channels  
        self.heads = heads 
        self.total_epochs = total_epochs

        self.model_transform = nn.Linear(model_in_channels, heads * hidden_channels, bias=False)
        self.target_transform = nn.Linear(target_in_channels, heads * hidden_channels, bias=False)

        self.edge_attn_transform = nn.Linear(768, heads * hidden_channels, bias=False)  

        self.gat_target = GATConv(heads * hidden_channels, hidden_channels, heads=heads, concat=True)

        self.batch_norm1 = nn.BatchNorm1d(2048, momentum=0.01)
        self.batch_norm2 = nn.BatchNorm1d(2048, momentum=0.01)
        
        self.prelu_m_t = nn.PReLU(num_parameters=heads)
        self.prelu_m_n = nn.PReLU(num_parameters=heads)
        self.prelu_t = nn.PReLU(num_parameters=heads)

        self.temperature_m_t = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.temperature_m_n = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.temperature_t = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        self.min_alpha, self.max_alpha = 0.05, 0.2  
        self.min_temp, self.max_temp = 0.5, 1.5  
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.model_transform.weight)
        nn.init.xavier_uniform_(self.target_transform.weight)
        nn.init.xavier_uniform_(self.edge_attn_transform.weight)

    def forward(self, model_x, target_x, edge_index, edge_attr):
        model_x_proj = self.model_transform(model_x).view(-1, self.heads * self.hidden_channels)
        target_x_proj = self.target_transform(target_x).view(-1, self.heads * self.hidden_channels)

        model_x_proj = self.batch_norm1(model_x_proj)
        target_x_proj = self.batch_norm2(target_x_proj)

        model_x_proj = model_x_proj.view(-1, self.heads, self.hidden_channels)
        target_x_proj = target_x_proj.view(-1, self.heads, self.hidden_channels)

        max_valid_index = target_x_proj.shape[0] - 1
        out_of_bounds_mask = edge_index[1] > max_valid_index

        edge_attr_proj = edge_attr[:, :768] - edge_attr[:, 768:]  

        edge_attn_values = self.edge_attn_transform(edge_attr_proj)  
        edge_attn_values = edge_attn_values.view(-1, self.heads, self.hidden_channels)  

        selected_target_x_proj = target_x_proj.index_select(0, edge_index[1])

        with torch.no_grad():  
            self.prelu_m_t.weight.data.clamp_(self.min_alpha, self.max_alpha)
            self.prelu_m_n.weight.data.clamp_(self.min_alpha, self.max_alpha)
            self.prelu_t.weight.data.clamp_(self.min_alpha, self.max_alpha)
            
        attn_m_t = torch.matmul(model_x_proj.index_select(0, edge_index[0]), edge_attn_values.transpose(-1, -2))  # (6001, 8, 8)
        attn_m_t = attn_m_t.mean(dim=-1)  # (6001, 8)
        attn_m_t = self.prelu_m_t(attn_m_t)
        attn_m_t = torch.softmax(attn_m_t / torch.clamp(self.temperature_m_t, self.min_temp, self.max_temp), dim=1)

        attn_m_n = torch.matmul(model_x_proj.index_select(0, edge_index[0]), edge_attn_values.transpose(-1, -2))
        attn_m_n = attn_m_n.mean(dim=-1)
        attn_m_n = self.prelu_m_n(attn_m_n)
        attn_m_n = torch.softmax(attn_m_n / torch.clamp(self.temperature_m_n, self.min_temp, self.max_temp), dim=1)

        attn_t = torch.matmul(target_x_proj.index_select(0, edge_index[1]), edge_attn_values.transpose(-1, -2))
        attn_t = attn_t.mean(dim=-1)
        attn_t = self.prelu_t(attn_t)
        attn_t = torch.softmax(attn_t / torch.clamp(self.temperature_t, self.min_temp, self.max_temp), dim=1)

        unique_target_nodes = edge_index[1].unique()
        target_x_proj = target_x_proj.view(edge_index.shape[1], 8, 256)
        edge_attn_values = edge_attn_values.view(edge_index.shape[1], 8, 256)

        target_x_new = (
            attn_m_n.unsqueeze(-1) * target_x_proj
            + attn_m_t.unsqueeze(-1) * edge_attn_values
            + attn_t.unsqueeze(-1) * target_x_proj
        )

        target_x_new = target_x_new.view(edge_index.shape[1], 2048)

        target_x_new = torch_scatter.scatter_mean(
            target_x_new.float(), edge_index[1], dim=0, dim_size=unique_target_nodes.shape[0]
        )

        attn_sum = attn_m_t + attn_m_n + attn_t 
        attn_score = torch_scatter.scatter_mean(attn_sum.float(), edge_index[1], dim=0, dim_size=unique_target_nodes.shape[0])

        target_index = unique_target_nodes.tolist()  

        target_index = torch.tensor(target_index, dtype=torch.long, device=target_x_new.device)

        return target_x_new, target_index, attn_score

class FeatureEnhancement(nn.Module):
    def __init__(self, in_channels=2048):
        super(FeatureEnhancement, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels)  
        self.bn1 = nn.BatchNorm1d(in_channels, momentum=0.01)  

    def forward(self, x):
        x = self.fc1(x)  
        x = self.bn1(x) 
        x = F.relu(x)  
        return x

class TargetFeatureEnhancement(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256):
        super(TargetFeatureEnhancement, self).__init__()
        self.fc1 = nn.Linear(in_channels, 1024)  
        self.bn1 = nn.BatchNorm1d(1024, momentum=0.01)  
        
        self.fc2 = nn.Linear(1024, 512)  
        self.bn2 = nn.BatchNorm1d(512, momentum=0.01)  
        
        self.fc3 = nn.Linear(512, out_channels)  
        self.bn3 = nn.BatchNorm1d(out_channels, momentum=0.01)  

    def forward(self, x):
        x = self.fc1(x)  
        x = self.bn1(x)  
        x = F.relu(x) 
        x = self.fc2(x)  
        x = self.bn2(x)  
        x = F.relu(x)  
        x = self.fc3(x) 
        x = self.bn3(x) 
        x = F.relu(x) 
        return x

class TargetDominationLayer(nn.Module):
    def __init__(self, hidden_channels, init_decay=0.8):
        super(TargetDominationLayer, self).__init__()

        self.edge_fc = nn.Linear(768, 256, bias=False)

        with torch.no_grad():
            Q, _ = torch.linalg.qr(torch.randn(768, 256))
            self.edge_fc.weight.copy_(Q.T)

        self.edge_fc.weight.requires_grad = False

        self.x_fc = nn.Linear(256, 256)
        self.x_bn = nn.BatchNorm1d(256, momentum=0.01)

    def forward(self, x, target_edge_index, target_index, target_x_new):
        x_j = x[target_index]

        mask = torch.isin(target_edge_index[1], target_index)

        sorted_mask_indices = torch.argsort(
            torch.searchsorted(target_index, target_edge_index[1][mask])
        )
        mask_sorted = mask.nonzero(as_tuple=True)[0][sorted_mask_indices]

        filtered_edge_index = target_edge_index[:, mask_sorted]

        target_ids = filtered_edge_index[1]
        source_ids = filtered_edge_index[0]

        x_i = target_x_new[source_ids]

        x_i_ave = torch_scatter.scatter_mean(
            x_i.float(),
            target_index[target_ids],
            dim=0,
            dim_size=x.shape[0]
        )

        x_diff = x_i_ave - x_j

        x_diff = self.x_fc(x_diff)
        x_diff = self.x_bn(x_diff)
        x_diff = F.relu(x_diff)

        memory_applied = x_diff + x_j

        return memory_applied, x_diff

class DominationLayer(nn.Module):
    def __init__(self, hidden_channels):
        super(DominationLayer, self).__init__()

        self.fc_dom_true = nn.Linear(768, 256, bias=False)

        with torch.no_grad():
            Q, _ = torch.linalg.qr(torch.randn(768, 256))
            self.fc_dom_true.weight.copy_(Q.T)

        self.fc_dom_true.weight.requires_grad = False

        self.fc1 = nn.Linear(hidden_channels, 256)
        self.bn1 = nn.BatchNorm1d(256, momentum=0.01)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256, momentum=0.01)

    def forward(self, memory_applied):
        dom_out = self.fc1(memory_applied)
        dom_out = self.bn1(dom_out)
        dom_out = F.relu(dom_out)

        dom_out = self.fc2(dom_out)
        dom_out = self.bn2(dom_out)
        dom_out = F.relu(dom_out)

        return dom_out

class GNNModel(nn.Module):
    def __init__(self, model_in_channels, target_in_channels, edge_in_channels, target_edge_in_channels, total_epochs):
        super(GNNModel, self).__init__()

        self.hidden_channels = 256
        self.heads = 8

        self.model_target_gat = ModelTargetAttentionGAT(
            model_in_channels,
            target_in_channels,
            edge_in_channels,
            self.hidden_channels,
            self.heads,
            total_epochs
        )

        self.feature_enhancement_target = FeatureEnhancement(self.heads * self.hidden_channels)

        self.target_feature_enhancement = TargetFeatureEnhancement(
            self.heads * self.hidden_channels,
            self.hidden_channels
        )

        self.target_domination = TargetDominationLayer(self.hidden_channels)

        self.domination_layer = DominationLayer(self.hidden_channels)

    def forward(
        self,
        model_x,
        target_x,
        model_target_edge_index,
        model_target_edge_attr,
        target_edge_index
    ):
        target_x_new, target_index, attn_score = self.model_target_gat(
            model_x,
            target_x,
            model_target_edge_index,
            model_target_edge_attr
        )

        target_x_new = self.feature_enhancement_target(target_x_new)

        target_x_new = self.target_feature_enhancement(target_x_new)

        memory_applied, x_diff = self.target_domination(
            target_x_new,
            target_edge_index,
            target_index,
            target_x_new
        )

        dom_out = self.domination_layer(memory_applied)

        return attn_score, memory_applied, x_diff, dom_out, target_index


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GNN_MODEL_PATH = "./data/gnn_data/200epoch/gnn_model_final.pt"

INPUT_ROOT_DIR = "./data/input_data/input_pt_data"
INPUT_PATTERN = "new_*"

EXPLOR_OUTPUT_ROOT = "./data/explor_data"
EVOLUTION_OUTPUT_ROOT = "./data/evolution_output"

PRIMARY_TOPK_MODEL_FEATURES = 10

PRIMARY_MAX_PRIMARY_NODES = 30

PRIMARY_SIM_TOL = 1e-6

USE_PATH_FILTERING = False

NGEN = 1
CXPB = 0.0
MUTPB = 0.2
SIMILARITY_THRESHOLD = 0.8


gnn_model = torch.load(GNN_MODEL_PATH, map_location=device)
gnn_model.eval()

torch.cuda.empty_cache()
gc.collect()

with torch.no_grad():
    attn_score, memory_applied, x_diff, dom_out, target_index = gnn_model(
        model_node_features,
        target_node_features,
        model_target_edge_index,
        model_target_edge_attr,
        target_edge_index
    )

del memory_applied
del x_diff

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("[LOG] GNN forward for exploration completed once.")
print(f"[LOG] attn_score shape: {attn_score.shape}")
print(f"[LOG] dom_out shape: {dom_out.shape}")
print(f"[LOG] target_index shape: {target_index.shape}")


def find_input_dirs(root_dir, pattern):
    import glob

    required_files = [
        "input_edge_index.pt",
        "input_edge_attr.pt",
        "input_target_edge_index.pt",
        "input_target_edge_attr.pt",
    ]

    dirs = sorted(glob.glob(os.path.join(root_dir, pattern)))
    valid_dirs = []

    for d in dirs:
        if not os.path.isdir(d):
            continue

        missing = [
            fname for fname in required_files
            if not os.path.exists(os.path.join(d, fname))
        ]

        if missing:
            print(f"[WARNING] Skipping {d}: missing files = {missing}")
            continue

        valid_dirs.append(d)

    print(f"[LOG] Found {len(valid_dirs)} valid input directories.")
    for d in valid_dirs:
        print(f"  - {d}")

    return valid_dirs


def load_input_pt_data(input_dir, device):
    input_edge_index = torch.load(os.path.join(input_dir, "input_edge_index.pt")).to(device)
    input_edge_attr = torch.load(os.path.join(input_dir, "input_edge_attr.pt")).to(device).float()
    input_target_edge_index = torch.load(os.path.join(input_dir, "input_target_edge_index.pt")).to(device)
    input_target_edge_attr = torch.load(os.path.join(input_dir, "input_target_edge_attr.pt")).to(device).float()

    return input_edge_index, input_edge_attr, input_target_edge_index, input_target_edge_attr


def select_primary_nodes_from_similar_model_target_edges(
    input_edge_attr,
    model_target_edge_index,
    model_target_edge_attr,
    attn_score,
    target_index,
    target_out_degree_dict,
    top_k=5,
    max_primary_nodes=20,
    sim_tol=1e-6
):

    print("[LOG] Entering select_primary_nodes_from_similar_model_target_edges")
    # print("[LOG] Primary selection mode: similarity-group top-k with max primary node cap")
    # print(f"[LOG] top_k similarity groups = {top_k}")
    # print(f"[LOG] max_primary_nodes = {max_primary_nodes}")
    # print(f"[LOG] sim_tol = {sim_tol}")

    input_source_features_ave = input_edge_attr[:, :768].mean(dim=0)  # (768,)

    candidate_model_features = model_target_edge_attr[:, :768]  # (E, 768)

    similarities = F.cosine_similarity(
        candidate_model_features,
        input_source_features_ave.unsqueeze(0),
        dim=1
    )

    sorted_vals, sorted_idx = torch.sort(similarities, descending=True)

    groups_edge_indices = []
    groups_sims = []

    current_sim = None
    current_group = []

    sorted_vals_cpu = sorted_vals.detach().cpu()
    sorted_idx_cpu = sorted_idx.detach().cpu()

    for val, idx in zip(sorted_vals_cpu, sorted_idx_cpu):
        val_f = float(val.item())
        idx_i = int(idx.item())

        if current_sim is None:
            current_sim = val_f
            current_group = [idx_i]
            continue

        if abs(val_f - current_sim) <= sim_tol:
            current_group.append(idx_i)
        else:
            groups_edge_indices.append(current_group)
            groups_sims.append(current_sim)

            current_sim = val_f
            current_group = [idx_i]

            if len(groups_edge_indices) >= top_k:
                break

    if current_group and len(groups_edge_indices) < top_k:
        groups_edge_indices.append(current_group)
        groups_sims.append(current_sim)

    if len(groups_edge_indices) == 0:
        raise RuntimeError(
            "No similar model-target edge groups were selected. "
            "Check model_target_edge_attr and input_edge_attr."
        )

    # print(f"[RESULT] Selected similar model-target edge groups: {len(groups_edge_indices)}")
    # for group_id, (edge_ids, group_sim) in enumerate(zip(groups_edge_indices, groups_sims)):
    #     print(
    #         f"  group={group_id}, "
    #         f"similarity={group_sim:.6f}, "
    #         f"edges={len(edge_ids)}"
    #     )

    target_id_to_attn_row = {
        int(node_id.item()): row_idx
        for row_idx, node_id in enumerate(target_index.detach().cpu())
    }

    selected_target_order = []
    selected_target_feature_lists = defaultdict(list)

    max_primary_nodes = int(max_primary_nodes)

    for group_rank, edge_ids in enumerate(groups_edge_indices):
        if len(selected_target_order) >= max_primary_nodes:
            break

        edge_ids_tensor = torch.tensor(
            edge_ids,
            dtype=torch.long,
            device=model_target_edge_index.device
        )

        group_src_ids = model_target_edge_index[0, edge_ids_tensor].detach().cpu().tolist()
        group_tgt_ids = model_target_edge_index[1, edge_ids_tensor].detach().cpu().tolist()

        src_to_local_rows = defaultdict(list)
        src_order = []

        for local_row, src_id in enumerate(group_src_ids):
            src_id = int(src_id)

            if src_id not in src_to_local_rows:
                src_order.append(src_id)

            src_to_local_rows[src_id].append(local_row)

        for src_id in src_order:
            if len(selected_target_order) >= max_primary_nodes:
                break

            local_rows = src_to_local_rows[src_id]

            target_to_edge_ids = defaultdict(list)

            for local_row in local_rows:
                edge_idx = int(edge_ids[local_row])
                tgt_id = int(group_tgt_ids[local_row])

                if tgt_id not in target_id_to_attn_row:
                    continue

                target_to_edge_ids[tgt_id].append(edge_idx)

            if len(target_to_edge_ids) == 0:
                continue

            sorted_targets = sorted(
                target_to_edge_ids.keys(),
                key=lambda tid: (
                    -int(target_out_degree_dict.get(int(tid), 0)),
                    int(tid)
                )
            )

            for tgt_id in sorted_targets:
                tgt_id = int(tgt_id)

                if tgt_id not in selected_target_feature_lists:
                    if len(selected_target_order) >= max_primary_nodes:
                        break
                    selected_target_order.append(tgt_id)

                edge_ids_for_target = target_to_edge_ids[tgt_id]
                edge_ids_for_target_tensor = torch.tensor(
                    edge_ids_for_target,
                    dtype=torch.long,
                    device=model_target_edge_attr.device
                )

                feature_mean_for_this_src = model_target_edge_attr[
                    edge_ids_for_target_tensor,
                    768:
                ].mean(dim=0)

                selected_target_feature_lists[tgt_id].append(feature_mean_for_this_src)

                # print(
                #     f"[PRIMARY] group={group_rank}, "
                #     f"src_model={src_id}, "
                #     f"target={tgt_id}, "
                #     f"target_out_degree={target_out_degree_dict.get(tgt_id, 0)}, "
                #     f"selected_count={len(selected_target_order)}"
                # )

    if len(selected_target_order) == 0:
        raise RuntimeError(
            "No primary target nodes were selected after max-node filtering. "
            "Check target_index consistency and selection thresholds."
        )

    unique_ids = torch.tensor(
        selected_target_order,
        dtype=torch.long,
        device=model_target_edge_index.device
    )

    mean_features = torch.stack([
        torch.stack(selected_target_feature_lists[int(tgt_id)], dim=0).mean(dim=0)
        for tgt_id in selected_target_order
    ], dim=0)

    print(f"[LOG] Selected unique primary target nodes: {len(selected_target_order)}")
    # print(f"[LOG] primary cap: {max_primary_nodes}")

    attn_rows = [
        target_id_to_attn_row[int(tgt_id)]
        for tgt_id in selected_target_order
    ]

    attn_rows_tensor = torch.tensor(
        attn_rows,
        dtype=torch.long,
        device=attn_score.device
    )

    primary_target_features_index = unique_ids.unsqueeze(1)  
    primary_target_features_attr = mean_features 

    # print(f"[LOG] primary_target_features_index shape: {primary_target_features_index.shape}")
    # print(f"[LOG] primary_target_features_attr shape: {primary_target_features_attr.shape}")

    assert primary_target_features_index.shape[1] == 1
    assert primary_target_features_attr.shape[1] == 768

    attn_weights = attn_score[attn_rows_tensor]

    if attn_weights.ndim == 2:
        attn_weights = attn_weights.mean(dim=1)

    primary_target_features_attr = primary_target_features_attr * attn_weights.unsqueeze(1)

    # print("[LOG] Primary features weighted by attention score.")
    # print("First vector (first 6 dims):", primary_target_features_attr[0][:6].detach().cpu().tolist())
    # print("Last vector (last 6 dims):", primary_target_features_attr[-1][-6:].detach().cpu().tolist())

    del candidate_model_features
    del similarities
    del sorted_vals
    del sorted_idx
    del sorted_vals_cpu
    del sorted_idx_cpu
    del groups_edge_indices
    del groups_sims

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return primary_target_features_index, primary_target_features_attr


def determine_directed_edges_from_dom_out(dom_out, target_index, target_edge_index_cpu, target_edge_attr_cpu):
    print("[LOG] Entering determine_directed_edges_from_dom_out")

    dom_out_cpu = dom_out.detach().cpu()
    target_index_cpu = target_index.detach().cpu().long()
    target_edge_index_cpu = target_edge_index_cpu.cpu().long()
    target_edge_attr_cpu = target_edge_attr_cpu.cpu().float()

    dom_strength_partial = torch.tensor([
        vec[vec > 0].mean().item() if (vec > 0).any() else 0.0
        for vec in dom_out_cpu
    ], dtype=torch.float32)

    num_nodes = int(target_edge_index_cpu.max().item()) + 1
    dom_strength_full = torch.zeros(num_nodes, dtype=torch.float32)
    dom_strength_full[target_index_cpu] = dom_strength_partial

    src_strength = dom_strength_full[target_edge_index_cpu[0]]
    tgt_strength = dom_strength_full[target_edge_index_cpu[1]]

    keep_mask = src_strength > tgt_strength

    keep_edges_index = target_edge_index_cpu[:, keep_mask].contiguous()
    keep_edges_attr = target_edge_attr_cpu[keep_mask].contiguous()

    # print(f"[LOG] keep_edges_index shape: {keep_edges_index.shape}")
    # print(f"[LOG] keep_edges_attr shape: {keep_edges_attr.shape}")

    return keep_edges_index, keep_edges_attr


def build_edge_lookup(keep_edges_index, keep_edges_attr):
    print("[LOG] Entering build_edge_lookup")
    edge_lookup = defaultdict(list)
    edge_attr_dict = defaultdict(list)

    for i in range(keep_edges_index.shape[1]):
        src = int(keep_edges_index[0, i].item())
        tgt = int(keep_edges_index[1, i].item())

        edge_lookup[(src, tgt)].append(i)
        edge_attr_dict[(src, tgt)].append(keep_edges_attr[i])  # tensor of shape (1536,)

    # print(f"[LOG] edge_lookup constructed: {len(edge_lookup)} unique (src, tgt) pairs")
    # for i, ((src, tgt), idx_list) in enumerate(edge_lookup.items()):
    #     print(f"  [edge_lookup] ({src}, {tgt}) -> edge indices: {idx_list}")
    #     if i >= 9:
    #         print("  ... (truncated)")
    #         break

    # print(f"[LOG] edge_attr_dict constructed: {len(edge_attr_dict)} unique (src, tgt) pairs")
    # for i, ((src, tgt), attr_list) in enumerate(edge_attr_dict.items()):
    #     print(f"  [edge_attr_dict] ({src}, {tgt}) -> attr[0][:3]: {attr_list[0][:3].tolist()} (1st only)")
    #     if i >= 9:
    #         print("  ... (truncated)")
    #         break

    return edge_lookup, edge_attr_dict


def edge_attr_dict_to_tensor(edge_attr_dict):
    start_time = time.time()
    edge_attr_list = []
    edge_index_list = []

    for (src, tgt), attr_list in edge_attr_dict.items():
        for attr in attr_list:
            edge_index_list.append([src, tgt])
            edge_attr_list.append(attr)

    edge_index = torch.tensor(edge_index_list).T  # shape: (2, E)
    edge_attr_tensor = torch.stack(edge_attr_list, dim=0)  # shape: (E, F)
    # print(f"[LOG] edge_attr_dict_to_tensor completed in {time.time() - start_time:.2f}s")
    return edge_index, edge_attr_tensor

def generate_paths_tensor_cpu(primary_index, input_edge_index, edge_index,
                              MAX_PATH_LENGTH=50, MAX_QUEUE_SIZE=1000000, GOAL_DISTANCE_THRESHOLD=2):
    print("[LOG] Entering generate_paths_tensor_cpu")
    goal_nodes = set(input_edge_index[1].unique().tolist())
    num_nodes = edge_index.max().item() + 1

    paths_by_start = defaultdict(list)

    for start_tensor in primary_index:
        start_node = int(start_tensor.item())
        queue = deque()
        queue.append((start_node, [], set()))  # (current_node, path, used_edges)
        visited_state = set()
        reached_goals = set()

        step = 0
        while queue:
            if len(queue) > MAX_QUEUE_SIZE:
                # print(f"[WARNING] Queue size exceeded limit at start node {start_node}. Forcing early termination.")
                break

            current_node, path, used_edges = queue.popleft()
            key = (current_node, frozenset(used_edges))
            if key in visited_state:
                continue
            visited_state.add(key)

            if len(path) >= MAX_PATH_LENGTH:
                continue

            mask = edge_index[0] == current_node
            next_edges = edge_index[:, mask]
            edge_indices = torch.nonzero(mask, as_tuple=False).squeeze()

            if next_edges.ndim == 1:
                next_edges = next_edges.unsqueeze(1)
            if edge_indices.ndim == 0:
                edge_indices = [edge_indices.item()]
            elif isinstance(edge_indices, torch.Tensor):
                edge_indices = edge_indices.tolist()

            for idx, (src, tgt) in zip(edge_indices, next_edges.T):
                src_item = src.item()
                tgt_item = tgt.item()

                if idx in used_edges:
                    continue

                if len(path) == 0:
                    if src_item != start_node:
                        continue
                else:
                    prev_edge = path[-1]
                    prev_tgt = edge_index[1, prev_edge].item()
                    if src_item != prev_tgt:
                        continue

                new_path = path + [idx]
                new_used = used_edges | {idx}
                queue.append((tgt_item, new_path, new_used))

                if tgt_item in goal_nodes and len(new_path) > 0:
                    try:
                        node_seq = [edge_index[0, new_path[0]].item()]
                        for e_idx in new_path:
                            node_seq.append(edge_index[1, e_idx].item())
                    except Exception as e:
                        # print(f"[WARNING] Failed to reconstruct node path: {e}")
                        continue

                    if node_seq[0] != start_node:
                        # print(f"[WARNING] Path ignored: node_seq[0] != start_node ({node_seq[0]} != {start_node})")
                        continue

                    if node_seq[-1] != tgt_item:
                        # print(f"[WARNING] Path ignored: node_seq[-1] != goal ({node_seq[-1]} != {tgt_item})")
                        continue

                    paths_by_start[start_node].append(new_path)
                    reached_goals.add(tgt_item)
                    # print(f"[TRACE] Verified goal path from {start_node}: {new_path}")

            step += 1
            if step % 100000 == 0 or not queue or step == 1:
                pass
                # print(f"[DEBUG] {start_node}: Step {step}, Queue size: {len(queue)}")

        total_edges = sum(len(path) for path in paths_by_start[start_node])
        used_nodes = set()
        for path in paths_by_start[start_node]:
            for idx in path:
                src = edge_index[0, idx].item()
                tgt = edge_index[1, idx].item()
                used_nodes.update([src, tgt])
        # print(f"[RESULT] Start node {start_node}: {len(paths_by_start[start_node])} paths, "
        #       f"{len(used_nodes)} nodes, {total_edges} total edges")

    print("[LOG] Path generation with CPU-based tensor traversal completed.")

    # print(f"[LOG] paths_by_start summary: {len(paths_by_start)} start nodes")
    # for i, (start, paths) in enumerate(paths_by_start.items()):
    #     print(f"  [paths_by_start] Start node: {start}, #paths: {len(paths)}")
    #     for j, path in enumerate(paths[:2]):
    #         try:
                # ノード列を reconstruct
    #             node_seq = [edge_index[0, path[0]].item()]
    #             for e_idx in path:
    #                 node_seq.append(edge_index[1, e_idx].item())
    #             node_seq_str = ', '.join(map(str, node_seq))
    #             print(f"    Path {j}: {path} | Nodes: [{node_seq_str}]")
    #         except Exception as e:
    #             print(f"    Path {j}: {path} | [ERROR reconstructing node path: {e}]")
    #     if i >= 4:
    #         print("  ... (truncated)")
    #         break
    
    return paths_by_start

def apply_paths_with_edge_trials_cpu(paths_by_start, primary_index, primary_attr,
                                     edge_index, edge_attr_tensor,
                                     input_edge_index, input_edge_attr,
                                     node_dim=768, threshold=1e-5, min_ratio=0.1):
    import time
    # print("[LOG] Starting apply_paths_with_edge_trials_cpu...")
    start_time = time.time()
    final_paths_by_start = {}

    flat_primary_index = [int(x) for x in primary_index.squeeze().tolist()]

    for i, start_tensor in enumerate(primary_index):
        start = int(start_tensor.item())
        start_feat = primary_attr[i].clone()

        paths = paths_by_start[start]
        valid_paths = []

        for edge_idx_path in paths:
            edge_feat_cache = {}
            node_state = {start: start_feat.clone()}
            partial_path = []

            for edge_idx in edge_idx_path:
                matches = [(src, tgt) for (src, tgt), idxs in edge_lookup.items() if edge_idx in idxs]
                if not matches:
                    break
                src, tgt = matches[0]

                if src not in node_state:
                    break  

                if (src, tgt, edge_idx) not in edge_feat_cache:
                    edge_feat_cache[(src, tgt, edge_idx)] = edge_attr_dict[(src, tgt)][
                        edge_lookup[(src, tgt)].index(edge_idx)][node_dim:]
                edge_feat = edge_feat_cache[(src, tgt, edge_idx)]

                delta = node_state[src] - edge_feat
                if tgt not in node_state:
                    node_state[tgt] = delta + edge_feat
                else:
                    node_state[tgt] += delta

                final_feat = node_state[tgt] + edge_feat
                active_dims = (final_feat.abs() > threshold).sum().item()
                if active_dims / final_feat.numel() < min_ratio:
                    break  

                partial_path.append(edge_idx)

            if partial_path:
                valid_paths.append(partial_path)

        if valid_paths:
            final_paths_by_start[start] = valid_paths

            total_edges = sum(len(path) for path in valid_paths)
            used_nodes = set()
            for path in valid_paths:
                for idx in path:
                    src = edge_index[0, idx].item()
                    tgt = edge_index[1, idx].item()
                    used_nodes.update([src, tgt])
            # print(f"[RESULT] Start node {start}: {len(valid_paths)} paths, "
            #       f"{len(used_nodes)} nodes, {total_edges} total edges")

    # print(f"[LOG] Path selection completed in {time.time() - start_time:.2f}s")
    return final_paths_by_start


keep_edges_index, keep_edges_attr = determine_directed_edges_from_dom_out(
    dom_out=dom_out,
    target_index=target_index,
    target_edge_index_cpu=target_edge_index_cpu,
    target_edge_attr_cpu=target_edge_attr
)

edge_lookup, edge_attr_dict = build_edge_lookup(
    keep_edges_index,
    keep_edges_attr
)

print("[LOG] Converting edge_attr_dict to tensor...")
edge_index, edge_attr_tensor = edge_attr_dict_to_tensor(edge_attr_dict)


def compute_total_loss_with_paths(final_paths_by_start, edge_index, edge_attr_tensor,
                                  primary_dict, input_edge_index, input_edge_attr, node_dim=768):
    print("[LOG] Entering compute_total_loss_with_paths")

    goal_index = input_edge_index[1]
    goal_features = input_edge_attr[:, node_dim:]
    node_goal_features = {
        int(goal_index[i].item()): goal_features[i]
        for i in range(goal_index.size(0))
    }

    pred_node_states = defaultdict(list)
    path_contributions = []

    for start_idx, (start, paths) in enumerate(final_paths_by_start.items(), 1):
        # print(f"[TRACE] Processing start node {start} ({start_idx}/{len(final_paths_by_start)}) with {len(paths)} paths")
        for path_idx, edge_idx_path in enumerate(paths):
            if not edge_idx_path:
                print(f"  [WARNING] Empty path at index {path_idx}, skipping.")
                continue

            # Start node consistency check
            first_edge = edge_idx_path[0]
            src_check = None
            src_check = int(edge_index[0, first_edge].item())

            if src_check != start:
                print(f"  [WARNING] Path {path_idx} does not start from start node {start}, but from {src_check}. Skipping.")
                continue

            edge_feat_cache = {}
            node_state = {}

            for i, edge_idx in enumerate(edge_idx_path):
                src = tgt = None
                try:
                    src = int(edge_index[0, edge_idx].item())
                    tgt = int(edge_index[1, edge_idx].item())
                except IndexError:
                    print(f"    [WARNING] Edge index {edge_idx} out of bounds in edge_index. Skipping.")
                    continue

                if i == 0:
                    if src in primary_dict:
                        node_state[src] = primary_dict[src].clone()
                    else:
                        print(f"    [WARNING] Source node {src} not in primary_dict. Skipping path.")
                        break
                elif src not in node_state:
                    print(f"    [WARNING] Source node {src} not in node_state during path. Skipping path.")
                    break

                if edge_idx not in edge_feat_cache:
                    edge_feat_cache[edge_idx] = edge_attr_tensor[edge_idx, node_dim:].to(device)
                edge_feat = edge_feat_cache[edge_idx]

                delta = node_state[src] - edge_feat
                updated_feat = delta + edge_feat
                node_state[tgt] = updated_feat

                # print(f"    [DEBUG] Step {i} | edge_idx: {edge_idx}, src: {src}, tgt: {tgt}")
                # print(f"    [DEBUG] node_state[src][:3]: {node_state[src][:3].tolist()}")
                # print(f"    [DEBUG] edge_feat[:3]: {edge_feat[:3].tolist()}")
                # print(f"    [DEBUG] delta[:3]: {delta[:3].tolist()}")
                # print(f"    [DEBUG] updated_feat[:3]: {updated_feat[:3].tolist()}")

            last_edge_idx = edge_idx_path[-1]
            final_tgt = None
            try:
                final_tgt = int(edge_index[1, last_edge_idx].item())
            except IndexError:
                print(f"  [WARNING] Final edge index {last_edge_idx} out of bounds. Skipping.")
                continue

            if final_tgt is not None and final_tgt in node_goal_features and final_tgt in node_state:
                pred_node_states[final_tgt].append(node_state[final_tgt])
                path_contributions.append((start, final_tgt, node_state[final_tgt], edge_idx_path))
            else:
                print(f"  [WARNING] Final target {final_tgt} not in goal or node_state. Skipping.")

    total_loss = 0.0
    for node, goal_feat in node_goal_features.items():
        if node in pred_node_states:
            pred_avg = torch.stack(pred_node_states[node], dim=0).mean(dim=0)
            loss = torch.norm(pred_avg - goal_feat, p=2)
        else:
            loss = torch.norm(torch.zeros_like(goal_feat) - goal_feat, p=2)
            print(f"[WARNING] Goal node {node} not reached. Using zero vector for loss.")
        # print(f"[RESULT] Loss for goal node {node}: {loss.item():.4f}")
        total_loss += loss.item()

    # print(f"[RESULT] Total loss: {total_loss:.4f}")
    
    # print("[LOG] Verifying contents of path_contributions...")
    # seen_starts = set()
    # printed = 0

    # for (start_node, final_tgt, feat, edge_idx_path) in path_contributions:
    #     if start_node not in seen_starts:
    #         seen_starts.add(start_node)
    #         printed = 0
    #     if printed >= 2:
    #         continue

    #     try:
    #         node_seq = [int(edge_index[0, edge_idx_path[0]].item())]
    #         for edge_idx in edge_idx_path:
    #             node_seq.append(int(edge_index[1, edge_idx].item()))
    #     except Exception as e:
    #         print(f"  [ERROR] Failed to reconstruct node sequence for path starting at {start_node}: {e}")
    #         continue

    #     print(f"  [PATH] Start: {start_node}, Goal: {final_tgt}, Nodes: {node_seq}, Path: {edge_idx_path}")
    #     printed += 1

    #     if len(seen_starts) >= 10:
    #         print("  ... (truncated)")
    #         break
    
    return total_loss, path_contributions, node_goal_features

def compute_coverage_from_paths(path_contributions, node_goal_features, total_loss, edge_index, total_coverage=1.0):
    print("[LOG] Entering compute_coverage_from_paths (CPU mode, path-wise coverage, raw delta loss)")

    epsilon = 1e-6
    node_to_feats = defaultdict(list)
    path_lookup = []

    for i, (start_node, final_tgt, final_feat, edge_idx_path) in enumerate(path_contributions):
        final_feat = final_feat.cpu()
        path_lookup.append((i, start_node, final_tgt, final_feat, edge_idx_path))
        node_to_feats[final_tgt].append(final_feat)

    node_goal_features = {k: v.cpu() for k, v in node_goal_features.items()}

    edge_contributions = {}

    for path_id, start_node, final_tgt, final_feat, edge_idx_path in tqdm(path_lookup, desc="Computing path-wise coverage"):
        temp_node_feats = {k: [f.clone() for f in v] for k, v in node_to_feats.items()}

        found = False
        for i, t in enumerate(temp_node_feats[final_tgt]):
            if torch.equal(final_feat, t):
                del temp_node_feats[final_tgt][i]
                found = True
                break
        if not found:
            print(f"[WARNING] Path {path_id} contribution not found in node {final_tgt}, skipping.")
            continue

        new_loss = 0.0
        for node, goal_feat in node_goal_features.items():
            feats = temp_node_feats.get(node, [])
            avg = torch.stack(feats, dim=0).mean(dim=0) if feats else torch.zeros_like(goal_feat)
            new_loss += torch.norm(avg - goal_feat, p=2).item()

        coverage = new_loss - total_loss
        edge_contributions[path_id] = coverage

    print("[LOG] Coverage computation (path-wise delta loss) completed.")

    top_paths = sorted(edge_contributions.items(), key=lambda x: x[1], reverse=True)[:10]
    print("[SUMMARY] Top 10 most beneficial paths (positive ΔLoss):")

    for path_id, cov in top_paths:
        try:
            _, start_node, final_tgt, _, edge_idx_path = path_lookup[path_id]

            node_seq = []
            if edge_idx_path:
                first_src = int(edge_index[0, edge_idx_path[0]].item())
                node_seq.append(first_src)
                for edge_idx in edge_idx_path:
                    tgt = int(edge_index[1, edge_idx].item())
                    node_seq.append(tgt)

            node_ids_str = ', '.join(map(str, node_seq))
            #ログ停止中 print(f"  Path ID: {path_id} | Start: {start_node} | Nodes: [{node_ids_str}] | ΔLoss: {cov:.4f}")

        except Exception as e:
            print(f"  [WARNING] Failed to log path {path_id}: {e}")

    return edge_contributions


if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))

if not hasattr(creator, "Individual"):
    creator.create("Individual", dict, fitness=creator.FitnessMulti)

def convert_paths_to_target_edges(final_paths_by_start, edge_index, target_edge_index):
    print("[LOG] Converting final_paths_by_start to target_edge_index-compatible format")

    original_edge_lookup = {}
    for idx in range(edge_index.size(1)):
        key = (int(edge_index[0, idx].item()), int(edge_index[1, idx].item()))
        original_edge_lookup[key] = idx

    target_lookup = defaultdict(list)
    for idx in range(target_edge_index.size(1)):
        key = (int(target_edge_index[0, idx].item()), int(target_edge_index[1, idx].item()))
        target_lookup[key].append(idx)

    converted_paths_by_start = defaultdict(list)

    for start_node, path_list in final_paths_by_start.items():
        for edge_path in path_list:
            target_edge_path = []
            for edge_idx in edge_path:
                # print(f"[DEBUG] Before int cast: edge_idx = {edge_idx}, type={type(edge_idx)}")
                edge_idx = int(edge_idx) if isinstance(edge_idx, torch.Tensor) else edge_idx  # ← ここ追加！
                src = int(edge_index[0, edge_idx].item())
                tgt = int(edge_index[1, edge_idx].item())
                key = (src, tgt)
                if key in target_lookup:
                    target_idx = int(target_lookup[key][0])  # ← ここもintに固定！
                    target_edge_path.append(target_idx)
                else:
                    print(f"[WARNING] No match in target_edge_index for ({src}->{tgt})")
                    break
            else:
                converted_paths_by_start[start_node].append(target_edge_path)

    print(f"[LOG] Conversion complete: {len(converted_paths_by_start)} start nodes")
    return converted_paths_by_start


def unify_paths_simple(converted_paths_by_start, max_logs=5):
    print("[LOG] Unifying paths by start node (simple deduplication)")

    unified_paths_by_start = defaultdict(list)

    for start_node, paths in converted_paths_by_start.items():
        seen_paths = set()

        for edge_path in paths:
            if not edge_path:
                continue
            path_key = tuple(edge_path)
            if path_key not in seen_paths:
                seen_paths.add(path_key)
                unified_paths_by_start[start_node].append(list(path_key))

    total_paths = sum(len(v) for v in unified_paths_by_start.values())
    print(f"[LOG] Unified paths generated: {total_paths} paths")

    # print("[LOG] Sample unified individuals:")
    # printed = 0
    # for start_node, paths in unified_paths_by_start.items():
    #     print(f"  [Start Node {start_node}] ({len(paths)} paths)")
    #     for i, path in enumerate(paths):
    #         print(f"    Path {i}: {path}")
    #         printed += 1
    #         if printed >= max_logs:
    #             print("  ... (truncated)")
    #             return unified_paths_by_start  

    return unified_paths_by_start


def assign_edge_choices(
    edge_seq,
    target_edge_index,
    target_edge_attr,
    primary_target_features_attr,
    edge_cand_dict,
    node_dim=768,
    similarity_threshold=0.8
):

    # print("[LOG] === assign_edge_choices START ===")
    device = target_edge_attr.device
    # print(f"[CHECK] target_edge_attr shape: {target_edge_attr.shape}")
    # print(f"[CHECK] example: {target_edge_attr[:5]}")  
    primary_target_features_attr = primary_target_features_attr.to(device)

    edge_choice = []
    prev_output_feat = None
    prev_node = None

    # print(f"[DEBUG] Incoming edge_seq: {edge_seq}")
    # print(f"[DEBUG] primary_target_features_attr shape: {primary_target_features_attr.shape}")

    i = 0
    while i < len(edge_seq):
        edge_idx = int(edge_seq[i])
        src = int(target_edge_index[0, edge_idx].item())
        tgt = int(target_edge_index[1, edge_idx].item())

        # print(f"[DEBUG] Step {i}: edge_idx={edge_idx}, src={src}, tgt={tgt}")

        key = (src, tgt)
        candidates = edge_cand_dict.get(key, [])

        # print(f"[DEBUG] Retrieved candidates for ({src}->{tgt}): {candidates}")
        # if candidates:
        #     for cand_idx in candidates:
        #         print(f"    [DEBUG] Candidate: {cand_idx}, type={type(cand_idx)}")
        # else:
        #     print(f"[WARNING] No candidates for key ({src}->{tgt})")

        if not candidates:
            print(f"[WARNING] No candidates for key ({src}->{tgt}), using edge_idx {edge_idx} directly.")
            edge_choice.append(edge_idx)
            prev_output_feat = target_edge_attr[edge_idx, node_dim:].to(device)
            prev_node = tgt
            i += 1
            continue

        if i == 0:
            # print(f"[DEBUG] First edge. Selecting based on primary_target_features_attr.")
            best_sim = -1.0
            best_cand_idx = candidates[0]
            for cand_edge_idx in candidates:
                cand_src_feat = target_edge_attr[cand_edge_idx, :node_dim].to(device)
                cos_sim = F.cosine_similarity(primary_target_features_attr, cand_src_feat, dim=0).item()
                # print(f"[DEBUG] Candidate edge {cand_edge_idx}: cos_sim={cos_sim:.6f}")
                if cos_sim > best_sim:
                    best_sim = cos_sim
                    best_cand_idx = cand_edge_idx
            # print(f"[DEBUG] Selected first edge: {best_cand_idx} with sim {best_sim:.6f}")
            edge_choice.append(best_cand_idx)
            prev_output_feat = target_edge_attr[best_cand_idx, node_dim:].to(device)
            prev_node = tgt
            i += 1
            continue

        group = []
        group_indices = []

        j = i
        while j < len(edge_seq):
            next_edge_idx = int(edge_seq[j])
            next_src = int(target_edge_index[0, next_edge_idx].item())
            if next_src != prev_node:
                break
            group.append(next_edge_idx)
            group_indices.append(j)
            j += 1

        if group:
            # print(f"[DEBUG] Found group from step {i} to {j-1} (same src={prev_node})")
            for idx_in_group, group_edge_idx in zip(group_indices, group):
                src = int(target_edge_index[0, group_edge_idx].item())
                tgt = int(target_edge_index[1, group_edge_idx].item())
                key = (src, tgt)
                candidates = edge_cand_dict.get(key, [])

                best_sim = -1.0
                best_cand_idx = candidates[0] if candidates else group_edge_idx
                for cand_edge_idx in candidates:
                    cand_src_feat = target_edge_attr[cand_edge_idx, :node_dim].to(device)
                    cos_sim = F.cosine_similarity(prev_output_feat, cand_src_feat, dim=0).item()
                    # print(f"[DEBUG] (Group) Candidate {cand_edge_idx}: cos_sim={cos_sim:.6f}")
                    if cos_sim > best_sim:
                        best_sim = cos_sim
                        best_cand_idx = cand_edge_idx

                # print(f"[DEBUG] (Group) Selected edge: {best_cand_idx} with sim {best_sim:.6f}")
                edge_choice.append(int(best_cand_idx))
                prev_output_feat = target_edge_attr[best_cand_idx, node_dim:].to(device)
                prev_node = tgt

            i = j
        else:
            candidates = edge_cand_dict.get((src, tgt), [])
            best_sim = -1.0
            best_cand_idx = candidates[0] if candidates else edge_idx
            for cand_edge_idx in candidates:
                cand_src_feat = target_edge_attr[cand_edge_idx, :node_dim].to(device)
                cos_sim = F.cosine_similarity(prev_output_feat, cand_src_feat, dim=0).item()
                # print(f"[DEBUG] (Isolated) Candidate {cand_edge_idx}: cos_sim={cos_sim:.6f}")
                if cos_sim > best_sim:
                    best_sim = cos_sim
                    best_cand_idx = cand_edge_idx

            # print(f"[DEBUG] (Isolated) Selected edge: {best_cand_idx} with sim {best_sim:.6f}")
            edge_choice.append(int(best_cand_idx))
            prev_output_feat = target_edge_attr[best_cand_idx, node_dim:].to(device)
            prev_node = tgt
            i += 1

    # print("[LOG] === assign_edge_choices END ===")
    # print(f"[LOG] Final edge_choice: {edge_choice}")
    return edge_choice

def separate_individuals_by_goal(unified_paths_by_start, target_edge_index, target_edge_attr, primary_dict, primary_target_features_attr, primary_target_features_index, edge_cand_dict, similarity_threshold=0.8):
    print("[LOG] Separating individuals by goal node (CPU enforced)")

    target_edge_index = target_edge_index.cpu()

    primary_nodes = primary_target_features_index.squeeze(1).cpu().numpy().tolist()
    start_node_to_row_idx = {node_id: idx for idx, node_id in enumerate(primary_nodes)}

    goal_to_individuals = defaultdict(list)

    for start_node, paths in unified_paths_by_start.items():
        if start_node not in primary_dict:
            print(f"[WARNING] Start node {start_node} missing in primary_dict.")
            continue
        if start_node not in start_node_to_row_idx:
            print(f"[WARNING] Start node {start_node} missing in primary_target_features_index.")
            continue

        row_idx = start_node_to_row_idx[start_node]
        primary_feat = primary_target_features_attr[row_idx]

        for edge_path in paths:
            # print(f"[DEBUG] Processing edge_path: {edge_path}")
            if not edge_path:
                continue
            try:
                last_edge = edge_path[-1]
                # print(f"[DEBUG] Last edge for final_tgt: {last_edge}, type={type(last_edge)}")
                final_tgt = int(target_edge_index[1, last_edge].item())

                edge_choice = assign_edge_choices(
                    edge_seq=edge_path,
                    target_edge_index=target_edge_index,
                    target_edge_attr=target_edge_attr,
                    primary_target_features_attr=primary_feat,
                    edge_cand_dict=edge_cand_dict,
                    node_dim=768,
                    similarity_threshold=similarity_threshold
                )

                ind = creator.Individual({
                    "start": start_node,
                    "goal": final_tgt,
                    "edge_seq": edge_path,
                    "edge_choice": edge_choice,
                    "direction": [True] * len(edge_path)  
                })

                goal_to_individuals[final_tgt].append(ind)

            except Exception as e:
                print(f"[WARNING] Failed to resolve final_tgt for path: {e}")

    # print(f"[LOG] Generated {len(goal_to_individuals)} goal groups with individuals")
    # total_individuals = sum(len(v) for v in goal_to_individuals.values())
    # print(f"[LOG] Total individuals created: {total_individuals}")

    # print("[LOG] Sample individuals:")

    # printed = 0
    # for goal, inds in goal_to_individuals.items():
    #     print(f"  [Goal Node {goal}] ({len(inds)} individuals)")
    #     for i, ind in enumerate(inds[:3]):  # 各ゴールノードから最大3件表示
    #         print(f"    Individual {i}: Start={ind['start']}, Edge Seq={ind['edge_seq']}, Edge Choice={ind['edge_choice']}")
    #     printed += len(inds)
    #     if printed >= 10: 
    #         print("  ... (truncated)")
    #         break
    
    return goal_to_individuals


def compute_activation(norm, alpha=2.0, beta=1.0):
    return 2 * torch.sigmoid(alpha * (norm - beta))

def evaluate_individual(individual, target_edge_index, target_edge_attr,
                        primary_dict, node_goal_features, node_dim=768, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    node_state = {}
    start_node = individual["start"]
    goal_node = individual["goal"]
    edge_seq = individual["edge_seq"]
    edge_choice = individual["edge_choice"]
    direction_flags = individual["direction"]

    # print(f"\n=== [START] Evaluating individual: start={start_node}, goal={goal_node} ===")

    first_edge_seq = edge_seq[0]
    first_direction = direction_flags[0]
    first_src = int(target_edge_index[0, first_edge_seq].item())
    first_tgt = int(target_edge_index[1, first_edge_seq].item())
    if not first_direction:
        first_src, first_tgt = first_tgt, first_src

    if first_src not in primary_dict:
        print(f"[WARNING] Start node {first_src} not found in primary_dict.")
        return 1e6

    node_state[first_src] = primary_dict[first_src].clone().to(device)
    # print(f"[INFO] Initial state of start node {first_src}: {node_state[first_src][:5]}...")

    for i in range(len(edge_seq)):
        try:
            seq_edge_idx = edge_seq[i]  
            actual_edge_idx = edge_choice[i] 

            direction = direction_flags[i]
            src = int(target_edge_index[0, seq_edge_idx].item())
            tgt = int(target_edge_index[1, seq_edge_idx].item())
            if not direction:
                src, tgt = tgt, src
                # print(f"[DEBUG] Reversed edge direction at index {i}: now src={src}, tgt={tgt}")

            if src not in node_state:
                # print(f"[DEBUG] Skipping edge {edge_idx}: src node {src} not in state")
                continue

            src_feat = node_state[src].clone()
            edge_feat = target_edge_attr[actual_edge_idx, node_dim:].to(device)
            norm = torch.norm(src_feat)
            activation = compute_activation(norm, alpha=2.0, beta=1.0)

            if tgt in node_state:
                node_state[tgt] *= activation
                # print(f"[INFO] Edge {i}: tgt {tgt} already in state, added delta.")
            else:
                updated_feat = edge_feat * activation
                node_state[tgt] = updated_feat.detach().clone()
                # print(f"[INFO] Edge {i}: src={src}, tgt={tgt}, edge_idx={edge_idx}")
                # print(f"       src_feat[:3]={src_feat[:3].tolist()}, edge_feat[:3]={edge_feat[:3].tolist()}")
                # print(f"       updated_feat[:3]={updated_feat[:3].tolist()}")

        except Exception as e:
            print(f"[WARNING] Message passing failed at edge {edge_choice[i]}: {e}")
            continue

    if goal_node in node_state and goal_node in node_goal_features:
        pred_feat = node_state[goal_node]
        true_feat = node_goal_features[goal_node].to(device)
        loss = torch.norm(pred_feat - true_feat, p=2).item()

        pred_feat_out = pred_feat.detach().clone()

        # print(f"[RESULT] Goal node reached. pred_feat[:5]={pred_feat[:5].tolist()}")
        # print(f"         true_feat[:5]={true_feat[:5].tolist()}")
        # print(f"         → Loss = {loss:.4f}")

    else:
        loss = 1e6
        pred_feat_out = None
        # print(f"[RESULT] Goal node {goal_node} not reached or no ground truth. → Loss = {loss:.1e}")

    return loss, pred_feat_out

def bfs_find_path_partial(src_node, tgt_node, target_edge_index, max_steps=100000):
    adj_list = defaultdict(list)
    for edge_idx in range(target_edge_index.size(1)):
        src = int(target_edge_index[0, edge_idx].item())
        tgt = int(target_edge_index[1, edge_idx].item())
        adj_list[src].append((tgt, edge_idx))

    queue = deque([(src_node, [])])
    visited = set()
    steps = 0

    while queue:
        if steps > max_steps:
            print(f"[WARNING] BFS exceeded {max_steps} steps without finding path from {src_node} to {tgt_node}")
            return None 
        steps += 1

        current_node, path_so_far = queue.popleft()

        if current_node == tgt_node:
            return path_so_far 

        if current_node in visited:
            continue
        visited.add(current_node)

        for next_node, edge_idx in adj_list.get(current_node, []):
            if next_node not in visited:
                queue.append((next_node, path_so_far + [edge_idx]))

    return None

def mutate_individual_with_path_repair(individual, target_edge_index, target_edge_attr, edge_cand_dict,
                                        primary_dict, similarity_threshold=0.8, node_dim=768, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    edge_seq = individual["edge_seq"]
    edge_choice = individual["edge_choice"]
    direction_flags = individual["direction"]

    mutated = False
    mutation_index = -1
    repair_occurred = False

    for i in range(len(edge_seq)):
        src = int(target_edge_index[0, edge_seq[i]].item())
        tgt = int(target_edge_index[1, edge_seq[i]].item())
        key = (src, tgt)

        if key not in edge_cand_dict or len(edge_cand_dict[key]) <= 1:
            continue

        current_feat = target_edge_attr[edge_seq[i], :node_dim].to(device)
        candidates = [idx for idx in edge_cand_dict[key] if idx != edge_seq[i]]
        random.shuffle(candidates)

        for cand_idx in candidates:
            cand_feat = target_edge_attr[cand_idx, :node_dim].to(device)
            cos_sim = F.cosine_similarity(current_feat, cand_feat, dim=0).item()
            if cos_sim >= similarity_threshold:
                edge_seq[i] = cand_idx
                mutation_index = i
                mutated = True
                break

    if not mutated:
        # print("[STATUS_point_mutation] Mutation failed (no edge replaced)")
        return False

    repaired_edge_seq = [edge_seq[0]]
    repaired_direction_flags = [True]
    repaired_edge_choice = [edge_choice[0]]

    current_node = int(target_edge_index[1, edge_seq[0]].item())
    repair_start_index = 1

    for i in range(1, len(edge_seq)):
        prev_edge = repaired_edge_seq[-1]
        prev_tgt = int(target_edge_index[1, prev_edge].item())

        this_edge = edge_seq[i]
        this_src = int(target_edge_index[0, this_edge].item())
        this_tgt = int(target_edge_index[1, this_edge].item())

        if prev_tgt != this_src:
            print(f"[WARNING] Disconnection detected at step {i}: {prev_tgt} -> {this_src}. Attempting repair...")
            repair_path = bfs_find_path_partial(prev_tgt, this_src, target_edge_index)
            if repair_path is None:
                print(f"[ERROR] Failed to repair path from {prev_tgt} to {this_src}.")
                return False
            repair_occurred = True

            for repair_edge_idx in repair_path:
                repair_src = int(target_edge_index[0, repair_edge_idx].item())
                repair_tgt = int(target_edge_index[1, repair_edge_idx].item())

                key = (repair_src, repair_tgt)
                candidates = edge_cand_dict.get(key, [])
                if not candidates:
                    key = (repair_tgt, repair_src)
                    candidates = edge_cand_dict.get(key, [])
                    direction = False
                else:
                    direction = True

                if candidates:
                    if len(repaired_edge_seq) > 0:
                        prev_edge_idx = repaired_edge_seq[-1]
                        prev_feat = target_edge_attr[prev_edge_idx, node_dim:].to(device)
                    else:
                        print("[ERROR] Broken path detected at first edge — no prior edge to base similarity on.")
                        return False
                    
                    similarities = [F.cosine_similarity(prev_feat, target_edge_attr[c, :node_dim].to(device), dim=0).item() for c in candidates]
                    above_threshold = [(j, sim) for j, sim in enumerate(similarities) if sim >= similarity_threshold]
                    if above_threshold:
                        chosen_idx = random.choice(above_threshold)[0]
                        chosen = candidates[chosen_idx]
                    else:
                        chosen_idx = int(torch.tensor(similarities).argmax().item())
                        chosen = candidates[chosen_idx]
                else:
                    chosen = 0

                # print(f"[DEBUG] Path repairing step {i}, key={key}, candidates={candidates}")
                # print(f"[DEBUG] Repaired edge_choice: {chosen}")

                repaired_edge_seq.append(repair_edge_idx)
                repaired_direction_flags.append(direction)
                repaired_edge_choice.append(chosen)

                prev_tgt = repair_tgt if direction else repair_src
                current_node = prev_tgt

            repair_start_index = i

        else:
            repaired_edge_seq.append(this_edge)
            repaired_direction_flags.append(True)
            repaired_edge_choice.append(edge_choice[i]) 
        current_node = this_tgt

    recalc_start = mutation_index + 1 if mutation_index >= 0 else i
    if recalc_start >= len(repaired_edge_seq):
        # print("[NOTE] No recalculation needed; recalc_start exceeds sequence length.")
        individual["edge_seq"] = repaired_edge_seq
        individual["direction"] = repaired_direction_flags
        individual["edge_choice"] = repaired_edge_choice
    else:
        prev_edge_idx = repaired_edge_seq[recalc_start - 1]
        prev_feat = target_edge_attr[prev_edge_idx, node_dim:].to(device)

        for i in range(recalc_start, len(repaired_edge_seq)):
            edge_idx = repaired_edge_seq[i]
            direction = repaired_direction_flags[i]

            src = int(target_edge_index[0, edge_idx].item())
            tgt = int(target_edge_index[1, edge_idx].item())
            key = (src, tgt) if direction else (tgt, src)
            candidates = edge_cand_dict.get(key, [])

            if candidates:
                similarities = [F.cosine_similarity(prev_feat, target_edge_attr[c, :node_dim].to(device), dim=0).item()
                                for c in candidates]
                above_threshold = [(j, sim) for j, sim in enumerate(similarities) if sim >= similarity_threshold]
                if above_threshold:
                    chosen_idx = random.choice(above_threshold)[0]
                    chosen = candidates[chosen_idx]
                else:
                    chosen_idx = int(torch.tensor(similarities).argmax().item())
                    chosen = candidates[chosen_idx]

            else:
                chosen = 0

            # print(f"[DEBUG] Recalculating step {i}, key={key}, candidates={candidates}")
            # print(f"[DEBUG] Recalculated edge_choice: {chosen}")

            repaired_edge_choice[i] = chosen
            prev_edge_idx = edge_idx 
            prev_feat = target_edge_attr[edge_idx, node_dim:].to(device)

        individual["edge_seq"] = repaired_edge_seq
        individual["direction"] = repaired_direction_flags
        individual["edge_choice"] = repaired_edge_choice

    # print("[REPAIR_point_mutation] Path repair completed successfully.")
    # print(f"[REPAIR_point_mutation] Final edge_seq: {individual['edge_seq']}")
    # print(f"[REPAIR_point_mutation] Final direction: {individual['direction']}")
    # print(f"[REPAIR_point_mutation] Final edge_choice: {individual['edge_choice']}")

    # if mutated and not repair_occurred:
    #     print("[STATUS_point_mutation] Mutation succeeded without repair")
    # elif not mutated and repair_occurred:
    #     print("[STATUS_point_mutation] Mutation failed, repair applied")
    # elif mutated and repair_occurred:
    #     print("[STATUS_point_mutation] Mutation succeeded with repair")

    def check_path_validity(seq, choice, directions, label):
        for i in range(len(seq) - 1):
            idx1 = seq[i]
            idx2 = choice[i + 1]
            tgt1 = int(target_edge_index[1, idx1].item()) if directions[i] else int(target_edge_index[0, idx1].item())
            src2 = int(target_edge_index[0, idx2].item()) if directions[i + 1] else int(target_edge_index[1, idx2].item())
            if tgt1 != src2:
                print(f"[ERROR] Invalid path in {label}: {tgt1} -> {src2} at step {i} -> {i+1}")
                return False
        return True

    is_seq_valid = check_path_validity(individual["edge_seq"], individual["edge_seq"], individual["direction"], "edge_seq")
    is_choice_valid = check_path_validity(individual["edge_seq"], individual["edge_choice"], individual["direction"], "edge_choice")

    start_node = individual["start"]
    goal_node = individual["goal"]
    first_edge = individual["edge_seq"][0]
    last_edge = individual["edge_choice"][-1]
    actual_start = int(target_edge_index[0, first_edge].item())
    actual_goal = int(target_edge_index[1, last_edge].item())

    if actual_start != start_node:
        print(f"[ERROR] Start node mismatch: expected {start_node}, got {actual_start}")
        return False
    if actual_goal != goal_node:
        print(f"[ERROR] Goal node mismatch: expected {goal_node}, got {actual_goal}")
        return False
    if not (is_seq_valid and is_choice_valid):
        return False

    # print("[VALIDATION_point_mutation] Final path successfully validated.")
    return True

def reroute_middle_segment_via_repair(individual, target_edge_index, target_edge_attr, edge_cand_dict, primary_dict,
                                      similarity_threshold=0.8, node_dim=768, bfs_find_path_partial=None, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    edge_seq = individual["edge_seq"]
    edge_choice = individual["edge_choice"]
    direction_flags = individual["direction"]

    if len(edge_seq) < 2:
        return False 

    first_edge = edge_seq[0]
    last_edge = edge_seq[-1]
    first_dir = direction_flags[0]
    last_dir = direction_flags[-1]
    first_choice = edge_choice[0]
    last_choice = edge_choice[-1]

    first_node = int(target_edge_index[1, first_edge].item()) if first_dir else int(target_edge_index[0, first_edge].item())
    last_node = int(target_edge_index[0, last_edge].item()) if last_dir else int(target_edge_index[1, last_edge].item())

    repair_path = bfs_find_path_partial(first_node, last_node, target_edge_index)
    if repair_path is None:
        return False 

    repaired_edge_seq = [first_edge]
    repaired_direction_flags = [first_dir]
    repaired_edge_choice = [first_choice]

    current_node = first_node
    prev_tgt = current_node

    for repair_step, repair_edge_idx in enumerate(repair_path):
        src = int(target_edge_index[0, repair_edge_idx].item())
        tgt = int(target_edge_index[1, repair_edge_idx].item())

        if src == current_node:
            direction = True
            next_node = tgt
        elif tgt == current_node:
            direction = False
            next_node = src
        else:
            return False  

        key = (src, tgt) if direction else (tgt, src)
        candidates = edge_cand_dict.get(key, [])
        if candidates:
            if len(repaired_edge_seq) > 0:
                prev_edge_idx = repaired_edge_seq[-1]
                prev_feat = target_edge_attr[prev_edge_idx, node_dim:].to(device)
            else:
                prev_feat = target_edge_attr[repair_edge_idx, :node_dim].to(device)
            similarities = [F.cosine_similarity(prev_feat, target_edge_attr[c, :node_dim].to(device), dim=0).item() for c in candidates]
            above_threshold = [(j, sim) for j, sim in enumerate(similarities) if sim >= similarity_threshold]
            if above_threshold:
                choice = random.choice(above_threshold)[0]
                choice_idx = candidates[choice]
            else:
                choice = int(torch.tensor(similarities).argmax().item())
                choice_idx = candidates[choice]
        else:
            choice_idx = 0

        # print(f"[DEBUG] Segment mutation recalculating step {repair_step}, key={key}, candidates={candidates}")
        # print(f"[DEBUG] Segment mutation recalculated edge_choice: {choice_idx}")

        repaired_edge_seq.append(repair_edge_idx)
        repaired_direction_flags.append(direction)
        repaired_edge_choice.append(choice_idx)

        current_node = next_node
        prev_tgt = current_node

    key = (int(target_edge_index[0, last_edge].item()), int(target_edge_index[1, last_edge].item())) if last_dir \
        else (int(target_edge_index[1, last_edge].item()), int(target_edge_index[0, last_edge].item()))
    candidates = edge_cand_dict.get(key, [])

    if candidates:
        prev_edge_idx = repaired_edge_seq[-1]
        prev_feat = target_edge_attr[prev_edge_idx, node_dim:].to(device)
        similarities = [F.cosine_similarity(prev_feat, target_edge_attr[c, :node_dim].to(device), dim=0).item() for c in candidates]
        above_threshold = [(j, sim) for j, sim in enumerate(similarities) if sim >= similarity_threshold]
        if above_threshold:
            choice = random.choice(above_threshold)[0]
            last_choice = candidates[choice]
        else:
            choice = int(torch.tensor(similarities).argmax().item())
            last_choice = candidates[choice]
    else:
        last_choice = 0

    # print(f"[DEBUG] Segment mutation final step, key={key}, candidates={candidates}")
    # print(f"[DEBUG] Segment mutation edge_choice: {last_choice}")

    repaired_edge_seq.append(last_edge)
    repaired_direction_flags.append(last_dir)
    repaired_edge_choice.append(last_choice)

    individual["edge_seq"] = repaired_edge_seq
    individual["direction"] = repaired_direction_flags
    individual["edge_choice"] = repaired_edge_choice

    # print("[REPAIR_segment_mutation] Middle segment rerouted and repaired successfully.")
    # print(f"[REPAIR_segment_mutation] Final edge_seq: {individual['edge_seq']}")
    # print(f"[REPAIR_segment_mutation] Final direction: {individual['direction']}")
    # print(f"[REPAIR_segment_mutation] Final edge_choice: {individual['edge_choice']}")

    def check_path_validity(seq, choice, directions, label):
        for i in range(len(seq) - 1):
            idx1 = seq[i]
            idx2 = choice[i + 1]
            tgt1 = int(target_edge_index[1, idx1].item()) if directions[i] else int(target_edge_index[0, idx1].item())
            src2 = int(target_edge_index[0, idx2].item()) if directions[i + 1] else int(target_edge_index[1, idx2].item())
            if tgt1 != src2:
                print(f"[ERROR] Invalid path in {label}: {tgt1} -> {src2} at step {i} -> {i+1}")
                return False
        return True

    is_seq_valid = check_path_validity(individual["edge_seq"], individual["edge_seq"], individual["direction"], "edge_seq")
    is_choice_valid = check_path_validity(individual["edge_seq"], individual["edge_choice"], individual["direction"], "edge_choice")

    start_node = individual["start"]
    goal_node = individual["goal"]
    first_edge = individual["edge_seq"][0]
    last_edge = individual["edge_choice"][-1]
    actual_start = int(target_edge_index[0, first_edge].item())
    actual_goal = int(target_edge_index[1, last_edge].item())

    if actual_start != start_node:
        print(f"[ERROR] Start node mismatch: expected {start_node}, got {actual_start}")
        return False
    if actual_goal != goal_node:
        print(f"[ERROR] Goal node mismatch: expected {goal_node}, got {actual_goal}")
        return False
    if not (is_seq_valid and is_choice_valid):
        return False

    #ログ停止中 print("[VALIDATION_segment_mutation] Final path successfully validated.")
    
    return True

def crossover_individuals(ind1, ind2, target_edge_index):
    seq1, seq2 = ind1["edge_seq"], ind2["edge_seq"]
    dir1, dir2 = ind1["direction"], ind2["direction"]
    choice1, choice2 = ind1["edge_choice"], ind2["edge_choice"]

    min_len = min(len(seq1), len(seq2))
    if min_len < 2:
        return ind1, ind2 

    cx_point = random.randint(1, min_len - 1)

    new_seq1 = seq1[:cx_point] + seq2[cx_point:]
    new_seq2 = seq2[:cx_point] + seq1[cx_point:]

    new_dir1 = dir1[:cx_point] + dir2[cx_point:]
    new_dir2 = dir2[:cx_point] + dir1[cx_point:]

    new_choice1 = choice1[:cx_point] + choice2[cx_point:]
    new_choice2 = choice2[:cx_point] + choice1[cx_point:]

    try:
        new_goal1 = int(target_edge_index[1, new_seq1[-1]].item()) if new_dir1[-1] else int(target_edge_index[0, new_seq1[-1]].item())
        new_goal2 = int(target_edge_index[1, new_seq2[-1]].item()) if new_dir2[-1] else int(target_edge_index[0, new_seq2[-1]].item())

        ind1["edge_seq"], ind1["direction"], ind1["edge_choice"], ind1["goal"] = new_seq1, new_dir1, new_choice1, new_goal1
        ind2["edge_seq"], ind2["direction"], ind2["edge_choice"], ind2["goal"] = new_seq2, new_dir2, new_choice2, new_goal2

    except Exception as e:
        print(f"[WARNING] Crossover failed: {e}")

    return ind1, ind2

def compute_diversity(individual, population, edge_usage_counts=None,
                      path_cache=None, length_cache=None):
    import numpy as np

    start = individual["start"]
    goal = individual["goal"]
    edge_seq = individual["edge_seq"]
    edge_choice = individual["edge_choice"]

    def extract_node_path(indiv):
        path = []
        for i, edge_idx in enumerate(indiv["edge_seq"]):
            src = int(target_edge_index[0, edge_idx].item())
            tgt = int(target_edge_index[1, edge_idx].item())
            path.append(src if indiv["direction"][i] else tgt)
        last_edge = indiv["edge_seq"][-1]
        last_tgt = int(target_edge_index[1, last_edge].item())
        path.append(last_tgt)
        return path

    indiv_id = id(individual)
    if path_cache and indiv_id in path_cache:
        this_path = path_cache[indiv_id]
        path_length = length_cache[indiv_id]
    else:
        this_path_list = extract_node_path(individual)
        this_path = set(this_path_list)
        path_length = len(this_path_list)

    path_scores = []
    start_diff = 0
    goal_diff = 0
    length_diffs = []

    for other in population:
        if other is individual:
            continue

        other_id = id(other)
        if path_cache and other_id in path_cache:
            other_path = path_cache[other_id]
            other_len = length_cache[other_id]
        else:
            other_path_list = extract_node_path(other)
            other_path = set(other_path_list)
            other_len = len(other_path_list)

        jaccard = 1.0 - len(this_path & other_path) / max(1, len(this_path | other_path))
        path_scores.append(jaccard)

        if other["start"] != start:
            start_diff += 1
        if other["goal"] != goal:
            goal_diff += 1

        length_diffs.append(abs(other_len - path_length))

    path_diversity = np.mean(path_scores) if path_scores else 0
    start_div = start_diff / max(1, len(population))
    goal_div = goal_diff / max(1, len(population))
    length_penalty = 1.0 / (1.0 + np.mean(length_diffs)) if length_diffs else 1.0

    edge_rarity_score = 0
    if edge_usage_counts:
        for e in edge_choice:
            freq = edge_usage_counts.get(e, 1)
            edge_rarity_score += 1.0 / freq
        edge_rarity_score /= len(edge_choice)

    total_diversity = (
        0.25 * path_diversity +
        0.15 * start_div +
        0.15 * goal_div +
        0.25 * length_penalty +
        0.20 * edge_rarity_score
    )

    return total_diversity

def evaluate_individual_with_diversity(individual, target_edge_index, target_edge_attr,
                                       primary_dict, node_goal_features,
                                       population=None, edge_usage_counts=None,
                                       node_dim=768, device=None):
    device = torch.device("cpu")

    loss = evaluate_individual(
        individual,
        target_edge_index=target_edge_index,
        target_edge_attr=target_edge_attr,
        primary_dict=primary_dict,
        node_goal_features=node_goal_features,
        node_dim=node_dim,
        device=device
    )

    diversity = compute_diversity(individual, population, edge_usage_counts=edge_usage_counts)

    return (loss, -diversity)

def find_best_combinations_by_goal_beam(population, node_goal_features, beam_k=5, max_steps=10, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    goal_to_individuals = {}
    for ind in population:
        if ind.get("pred_feat") is not None:
            goal_to_individuals.setdefault(ind["goal"], []).append(ind)

    best_combinations = {}

    for goal, inds in goal_to_individuals.items():
        true_feat = node_goal_features.get(goal)
        if true_feat is None or len(inds) == 0:
            continue

        true_feat = true_feat.to(device)
        pred_feats = [ind["pred_feat"].to(device) for ind in inds]
        inds_tensor = list(inds)

        # print(f"[INFO] Processing goal node: {goal} (candidates: {len(pred_feats)})")
        # print(f"[INFO] true_feat shape: {true_feat.shape}")

        beam = [([i], pred_feats[i]) for i in range(len(pred_feats))]  # ([index_list], sum_tensor)
        best_combo = None
        best_dist = float('inf')

        for step in range(1, max_steps + 1):
            candidates = []

            # print(f"[STEP {step}] Beam size = {len(beam)}")

            for idx_list, sum_tensor in beam:
                used_set = set(idx_list)
                for j in range(len(pred_feats)):
                    if j in used_set:
                        continue
                    new_list = idx_list + [j]
                    new_sum = sum_tensor + pred_feats[j]
                    mean_feat = new_sum / len(new_list)

                    dist = F.pairwise_distance(mean_feat.unsqueeze(0), true_feat.unsqueeze(0), p=2).item()
                    candidates.append((new_list, new_sum, dist))

            # print(f"[LOG] Step {step} - top 5 candidates:")
            for k, (idxs, sum_tensor, dist_val) in enumerate(candidates[:5]):
                mean_feat = sum_tensor / len(idxs)
                # print(f"  [{k+1}] +{idxs[-1]} -> dist={dist_val:.6f} | mean_feat.shape={mean_feat.shape}, sum_tensor.shape={sum_tensor.shape}")

            if not candidates:
                break

            candidates.sort(key=lambda x: x[2])
            beam = [(c[0], c[1]) for c in candidates[:beam_k]]

            # print(f"[INFO] Beam updated: top-{beam_k} candidates retained.")

            top_combo, _, top_dist = candidates[0]
            if top_dist < best_dist:
                best_combo = top_combo
                best_dist = top_dist

        if best_combo is not None:
            best_combinations[goal] = {
                "combination": [inds_tensor[i] for i in best_combo],
                "distance": best_dist
            }
            print(f"[COMBO] Goal={goal} | Best combination length={len(best_combo)} | Distance={best_dist:.6f}")

    return best_combinations

def build_edge_usage_counts(population):
    all_edges = []
    for ind in population:
        all_edges.extend(ind["edge_choice"])
    return Counter(all_edges)


def tensor_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(tensor_to_serializable(v) for v in obj)
    return obj


def make_sample_save_dirs(sample_name):
    explor_save_dir = os.path.join(EXPLOR_OUTPUT_ROOT, sample_name)
    evolution_save_dir = os.path.join(EVOLUTION_OUTPUT_ROOT, sample_name)

    os.makedirs(explor_save_dir, exist_ok=True)
    os.makedirs(evolution_save_dir, exist_ok=True)

    return explor_save_dir, evolution_save_dir


def process_one_input(input_dir):
    sample_name = os.path.basename(os.path.normpath(input_dir))
    explor_save_dir, evolution_save_dir = make_sample_save_dirs(sample_name)

    print("\n" + "=" * 100)
    print(f"[LOG] Processing sample: {sample_name}")
    print(f"[LOG] input_dir: {input_dir}")
    print(f"[LOG] explor_save_dir: {explor_save_dir}")
    print(f"[LOG] evolution_save_dir: {evolution_save_dir}")
    print("=" * 100)

    input_edge_index, input_edge_attr, input_target_edge_index, input_target_edge_attr = load_input_pt_data(
        input_dir=input_dir,
        device=device
    )

    primary_target_features_index, primary_target_features_attr = select_primary_nodes_from_similar_model_target_edges(
        input_edge_attr=input_edge_attr,
        model_target_edge_index=model_target_edge_index,
        model_target_edge_attr=model_target_edge_attr,
        attn_score=attn_score,
        target_index=target_index,
        target_out_degree_dict=TARGET_OUT_DEGREE_DICT,
        top_k=PRIMARY_TOPK_MODEL_FEATURES,
        max_primary_nodes=PRIMARY_MAX_PRIMARY_NODES,
        sim_tol=PRIMARY_SIM_TOL
    )

    # print("[LOG] Generating paths using tensor-based CPU traversal...")
    paths_by_start = generate_paths_tensor_cpu(
        primary_index=primary_target_features_index,
        input_edge_index=input_edge_index,
        edge_index=edge_index
    )

    if USE_PATH_FILTERING:
        # print("[LOG] Applying CPU-based path evaluation...")
        final_paths_by_start = apply_paths_with_edge_trials_cpu(
            paths_by_start=paths_by_start,
            primary_index=primary_target_features_index,
            primary_attr=primary_target_features_attr,
            edge_index=edge_index,
            edge_attr_tensor=edge_attr_tensor,
            input_edge_index=input_edge_index,
            input_edge_attr=input_edge_attr
        )
        # print("[LOG] CPU path evaluation completed.")
    else:
        # print("[LOG] Skipping path filtering, using raw paths_by_start")
        final_paths_by_start = paths_by_start

    # for start_node in list(paths_by_start.keys())[:10]:
    #     print(f"[CHECK] Start node {start_node}")
    #     print(f"  - Total paths before filtering: {len(paths_by_start[start_node])}")
    #     print(f"  - Total paths after filtering: {len(final_paths_by_start.get(start_node, []))}")

    primary_dict = {
        int(primary_target_features_index[i].item()): primary_target_features_attr[i]
        for i in range(primary_target_features_index.size(0))
    }

    goal_index = input_edge_index[1]
    goal_features = input_edge_attr[:, 768:]

    node_goal_features = {
        int(goal_index[i].item()): goal_features[i]
        for i in range(goal_index.size(0))
    }

    total_loss, path_contributions, node_goal_features = compute_total_loss_with_paths(
        final_paths_by_start,
        edge_index,
        edge_attr_tensor,
        primary_dict,
        input_edge_index,
        input_edge_attr
    )

    delta_loss = {}

    with open(os.path.join(explor_save_dir, "found_paths.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in final_paths_by_start.items()}, f, indent=2)

    with open(os.path.join(explor_save_dir, "delta_loss.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in delta_loss.items()}, f, indent=2)

    torch.save(final_paths_by_start, os.path.join(explor_save_dir, "found_paths.pt"))
    torch.save(delta_loss, os.path.join(explor_save_dir, "delta_loss.pt"))

    path_contributions_serializable = [
        {
            "start_node": int(start_node),
            "final_tgt": int(final_tgt),
            "edge_idx_path": [int(e) for e in edge_idx_path]
        }
        for (start_node, final_tgt, feat, edge_idx_path) in path_contributions
    ]

    with open(os.path.join(explor_save_dir, "path_contributions.json"), "w", encoding="utf-8") as f:
        json.dump(path_contributions_serializable, f, indent=2)

    torch.save(primary_target_features_index, os.path.join(explor_save_dir, "primary_target_features_index.pt"))
    torch.save(primary_target_features_attr, os.path.join(explor_save_dir, "primary_target_features_attr.pt"))

    print(f"[LOG] Saved exploration results to {explor_save_dir}")

    converted_paths_by_start = convert_paths_to_target_edges(
        final_paths_by_start=final_paths_by_start,
        edge_index=edge_index,
        target_edge_index=target_edge_index
    )

    unified_paths_by_start = unify_paths_simple(
        converted_paths_by_start=converted_paths_by_start
    )

    edge_cand_dict = defaultdict(list)
    for idx in range(target_edge_index.size(1)):
        src = int(target_edge_index[0, idx].item())
        tgt = int(target_edge_index[1, idx].item())
        edge_cand_dict[(src, tgt)].append(idx)

    individuals_by_goal = separate_individuals_by_goal(
        unified_paths_by_start=unified_paths_by_start,
        target_edge_index=target_edge_index,
        target_edge_attr=target_edge_attr,
        primary_dict=primary_dict,
        primary_target_features_attr=primary_target_features_attr,
        primary_target_features_index=primary_target_features_index,
        edge_cand_dict=edge_cand_dict,
        similarity_threshold=SIMILARITY_THRESHOLD
    )

    toolbox = base.Toolbox()

    toolbox.register(
        "evaluate",
        evaluate_individual,
        target_edge_index=target_edge_index,
        target_edge_attr=target_edge_attr,
        primary_dict=primary_dict,
        node_goal_features=node_goal_features,
        node_dim=768,
        device=device
    )

    toolbox.register(
        "mutate",
        mutate_individual_with_path_repair,
        target_edge_index=target_edge_index,
        target_edge_attr=target_edge_attr,
        edge_cand_dict=edge_cand_dict,
        primary_dict=primary_dict,
        similarity_threshold=SIMILARITY_THRESHOLD,
        node_dim=768,
        device=device
    )

    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", crossover_individuals, target_edge_index=target_edge_index)

    population = [
        ind
        for individuals in individuals_by_goal.values()
        for ind in individuals
    ]

    print(f"[LOG] Total initial individuals: {len(population)}")

    if len(population) == 0:
        print(f"[WARNING] No individuals generated for {sample_name}. Skipping evolution.")
        return

    latest_population_info = []
    best_combinations = {}

    for gen in range(1, NGEN + 1):
        print(f"[GENERATION {gen}]")

        fitnesses = []
        for ind in population:
            loss, pred_feat_out = toolbox.evaluate(ind)
            fitnesses.append(loss)
            ind["pred_feat"] = pred_feat_out.detach() if pred_feat_out is not None else None

        path_cache = {}
        length_cache = {}

        for ind in population:
            path = []
            for i, edge_idx in enumerate(ind["edge_seq"]):
                src = int(target_edge_index[0, edge_idx].item())
                tgt = int(target_edge_index[1, edge_idx].item())
                path.append(src if ind["direction"][i] else tgt)
            last_edge = ind["edge_seq"][-1]
            path.append(int(target_edge_index[1, last_edge].item()))
            path_cache[id(ind)] = set(path)
            length_cache[id(ind)] = len(path)

        edge_usage_counts = build_edge_usage_counts(population)

        for ind, loss in zip(population, fitnesses):
            diversity = compute_diversity(
                ind,
                population,
                edge_usage_counts=edge_usage_counts,
                path_cache=path_cache,
                length_cache=length_cache
            )
            ind.fitness.values = (loss, -diversity)

        fitness_values = [ind.fitness.values[0] for ind in population]
        avg_fitness = sum(fitness_values) / len(fitness_values)
        min_fitness = min(fitness_values)
        print(f"[STATS] Avg fitness: {avg_fitness:.6f} | Min fitness: {min_fitness:.6f}")

        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        new_offspring = []

        temp_fitnesses = [
            ind.fitness.values[0] if ind.fitness.valid else toolbox.evaluate(ind)[0]
            for ind in offspring
        ]

        fitness_threshold = sorted(temp_fitnesses)[int(0.8 * len(temp_fitnesses))] if temp_fitnesses else float("inf")

        for mutant, temp_fit in zip(offspring, temp_fitnesses):
            if random.random() < MUTPB:
                original = toolbox.clone(mutant)
                success = toolbox.mutate(mutant)

                if success:
                    loss, pred_feat_out = toolbox.evaluate(mutant)
                    mutant["pred_feat"] = pred_feat_out.detach() if pred_feat_out is not None else None
                    mutant.fitness.values = (loss, -1.0)
                    new_offspring.append(mutant)

                else:
                    if temp_fit > fitness_threshold and random.random() < 0.5:
                        repaired = reroute_middle_segment_via_repair(
                            individual=mutant,
                            target_edge_index=target_edge_index,
                            target_edge_attr=target_edge_attr,
                            edge_cand_dict=edge_cand_dict,
                            primary_dict=primary_dict,
                            similarity_threshold=SIMILARITY_THRESHOLD,
                            node_dim=768,
                            bfs_find_path_partial=bfs_find_path_partial,
                            device=device
                        )

                        if repaired:
                            loss, pred_feat_out = toolbox.evaluate(mutant)
                            mutant["pred_feat"] = pred_feat_out.detach() if pred_feat_out is not None else None
                            mutant.fitness.values = (loss, -1.0)
                            new_offspring.append(mutant)
                            continue

                    loss, pred_feat_out = toolbox.evaluate(original)
                    original["pred_feat"] = pred_feat_out.detach() if pred_feat_out is not None else None
                    original.fitness.values = (loss, -1.0)
                    new_offspring.append(original)

            else:
                loss, pred_feat_out = toolbox.evaluate(mutant)
                mutant["pred_feat"] = pred_feat_out.detach() if pred_feat_out is not None else None
                mutant.fitness.values = (loss, -1.0)
                new_offspring.append(mutant)

        population[:] = new_offspring

        path_cache = {}
        length_cache = {}

        for ind in population:
            path = []
            for i, edge_idx in enumerate(ind["edge_seq"]):
                src = int(target_edge_index[0, edge_idx].item())
                tgt = int(target_edge_index[1, edge_idx].item())
                path.append(src if ind["direction"][i] else tgt)
            last_edge = ind["edge_seq"][-1]
            path.append(int(target_edge_index[1, last_edge].item()))
            path_cache[id(ind)] = set(path)
            length_cache[id(ind)] = len(path)

        edge_usage_counts = build_edge_usage_counts(population)

        for ind in population:
            loss, pred_feat_out = toolbox.evaluate(ind)
            ind["pred_feat"] = pred_feat_out.detach() if pred_feat_out is not None else None
            diversity = compute_diversity(
                ind,
                population,
                edge_usage_counts=edge_usage_counts,
                path_cache=path_cache,
                length_cache=length_cache
            )
            ind.fitness.values = (loss, -diversity)

        best_combinations = find_best_combinations_by_goal_beam(
            population=population,
            node_goal_features=node_goal_features,
            beam_k=50,
            max_steps=20,
            device=device
        )

        latest_population_info = [(ind, ind.fitness.values) for ind in population]

        best_fit = min(fit[0] for _, fit in latest_population_info)
        # print(f"[RESULT] Best fitness at generation {gen}: {best_fit:.6f}")

    print("[LOG] Evolution completed.")

    if len(latest_population_info) == 0:
        print(f"[WARNING] latest_population_info is empty for {sample_name}.")
        return

    best_individual, best_fitness = min(latest_population_info, key=lambda x: x[1])

    best_individual_dict = dict(best_individual)
    best_individual_dict["fitness"] = tensor_to_serializable(best_fitness)

    with open(os.path.join(evolution_save_dir, "best_individual.json"), "w", encoding="utf-8") as f:
        json.dump(tensor_to_serializable(best_individual_dict), f, indent=2)

    with open(os.path.join(evolution_save_dir, "best_fitness.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best Fitness (Final Loss): {best_fitness[0]:.6f}, Diversity: {-best_fitness[1]:.6f}\n")

    torch.save(best_individual_dict, os.path.join(evolution_save_dir, "best_individual.pt"))

    all_individuals_serialized = []
    for ind, fit in latest_population_info:
        ind_dict = dict(ind)
        ind_dict["fitness"] = tensor_to_serializable(fit)
        all_individuals_serialized.append(tensor_to_serializable(ind_dict))

    with open(os.path.join(evolution_save_dir, "all_individuals.json"), "w", encoding="utf-8") as f:
        json.dump(all_individuals_serialized, f, indent=2)

    best_individuals_by_pair = {}

    for ind, fit in latest_population_info:
        start = ind["start"]
        goal = ind["goal"]
        key = (start, goal)

        if key not in best_individuals_by_pair or fit < best_individuals_by_pair[key][1]:
            best_individuals_by_pair[key] = (ind, fit)

    best_individuals_by_pair_serialized = {}

    for (start, goal), (best_ind, fit) in best_individuals_by_pair.items():
        key_str = f"start{start}_goal{goal}"
        ind_dict = dict(best_ind)
        ind_dict["fitness"] = tensor_to_serializable(fit)
        best_individuals_by_pair_serialized[key_str] = tensor_to_serializable(ind_dict)

    with open(os.path.join(evolution_save_dir, "best_individuals_by_pair.json"), "w", encoding="utf-8") as f:
        json.dump(best_individuals_by_pair_serialized, f, indent=2)

    torch.save(best_individuals_by_pair_serialized, os.path.join(evolution_save_dir, "best_individuals_by_pair.pt"))

    best_combinations_serialized = {}

    for goal, combo_info in best_combinations.items():
        serialized = {
            "goal": goal,
            "distance": combo_info["distance"],
            "combination": [
                tensor_to_serializable(dict(ind))
                for ind in combo_info["combination"]
            ]
        }
        best_combinations_serialized[str(goal)] = serialized

    with open(os.path.join(evolution_save_dir, "best_combinations_by_goal.json"), "w", encoding="utf-8") as f:
        json.dump(best_combinations_serialized, f, indent=2)

    sample_summary = {
        "sample_name": sample_name,
        "input_dir": input_dir,
        "explor_save_dir": explor_save_dir,
        "evolution_save_dir": evolution_save_dir,
        "num_primary_nodes": int(primary_target_features_index.shape[0]),
        "num_start_nodes_with_paths": int(len(final_paths_by_start)),
        "total_loss": float(total_loss),
        "n_population_final": int(len(population)),
        "n_generation": int(NGEN),
        "primary_topk_model_features": int(PRIMARY_TOPK_MODEL_FEATURES),
        "primary_max_primary_nodes": int(PRIMARY_MAX_PRIMARY_NODES),
        "primary_similarity_tolerance": float(PRIMARY_SIM_TOL),
        "primary_selection_method": "top_k_similarity_groups_with_primary_cap_and_target_out_degree_priority",
    }

    with open(os.path.join(evolution_save_dir, "sample_summary.json"), "w", encoding="utf-8") as f:
        json.dump(sample_summary, f, indent=2)

    print(f"[LOG] Evolution outputs saved to {evolution_save_dir}")

    del input_edge_index
    del input_edge_attr
    del input_target_edge_index
    del input_target_edge_attr
    del primary_target_features_index
    del primary_target_features_attr
    del paths_by_start
    del final_paths_by_start
    del primary_dict
    del node_goal_features
    del path_contributions
    del population
    del latest_population_info

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


input_dirs = find_input_dirs(
    root_dir=INPUT_ROOT_DIR,
    pattern=INPUT_PATTERN
)

if not input_dirs:
    raise RuntimeError(
        f"No valid input directories found under {INPUT_ROOT_DIR} with pattern {INPUT_PATTERN}"
    )

for input_dir in input_dirs:
    try:
        process_one_input(input_dir)
    except Exception as e:
        print(f"[ERROR] Failed processing {input_dir}: {e}")
        raise

print("\nAll input samples processed.")
