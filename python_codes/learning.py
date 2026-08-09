import os
import csv
import gc

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn import GATConv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_node_features = torch.load("./data/pt_data/model_features.pt")

model_target_edge_index = torch.load("./data/pt_data/edge_index.pt").to(device)
model_target_edge_attr = torch.load("./data/pt_data/edge_attr.pt").to(device).float()  # float32 に統一

target_node_features = torch.load("./data/pt_data/target_features.pt")

target_edge_index_cpu = torch.load("./data/pt_data/target_edge_index.pt").long()
target_edge_index = target_edge_index_cpu.to(device)

target_edge_attr_cpu = torch.load("./data/pt_data/target_edge_attr.pt").float()

target_edge_attr = None


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

output_dir = "./data/gnn_data/"
os.makedirs(output_dir, exist_ok=True)

loss_records = []

edge_weight_list = []
edge_weight_index_list = []

model_node_features = get_model_features(model_target_edge_index, model_node_features)

target_node_features = get_target_features(model_target_edge_index, target_node_features)

model_in_channels = model_node_features.shape[1]
target_in_channels = target_node_features.shape[1]
edge_in_channels = model_target_edge_attr.shape[1]

target_edge_in_channels = target_edge_attr_cpu.shape[1]

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
            
        attn_m_t = torch.matmul(model_x_proj.index_select(0, edge_index[0]), edge_attn_values.transpose(-1, -2))  
        attn_m_t = attn_m_t.mean(dim=-1)  
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

    def compute_fixed_edge_diff_cpu(
        self,
        target_index,
        target_edge_index_cpu,
        target_edge_attr_cpu,
        node_dim=768,
        chunk_size=200_000,
        output_device=None,
        max_dense_map_size=50_000_000
    ):

        print("[LOG] Computing fixed_edge_diff on CPU...")

        if output_device is None:
            output_device = next(self.parameters()).device

        target_index_cpu = target_index.detach().cpu().long()
        fixed_edge_diff_target_index_cpu = target_index_cpu.clone()

        target_edge_index_cpu = target_edge_index_cpu.cpu().long()
        target_edge_attr_cpu = target_edge_attr_cpu.cpu().float()

        num_targets = int(target_index_cpu.numel())
        total_edges = target_edge_index_cpu.size(1)

        if num_targets == 0:
            raise RuntimeError("target_index is empty. Cannot compute fixed_edge_diff.")

        max_target_id_in_index = int(target_index_cpu.max().item())
        max_target_id_in_edges = int(target_edge_index_cpu[1].max().item())
        max_target_id = max(max_target_id_in_index, max_target_id_in_edges)

        use_dense_map = max_target_id <= max_dense_map_size

        if use_dense_map:
            id_to_pos = torch.full(
                (max_target_id + 1,),
                -1,
                dtype=torch.long,
                device="cpu"
            )

            id_to_pos[target_index_cpu] = torch.arange(
                num_targets,
                dtype=torch.long,
                device="cpu"
            )

            id_to_pos_dict = None
            print(f"[LOG] Using dense target_id-to-position map: size={max_target_id + 1}")

        else:
            # target ID が非常に疎な場合は、巨大dense tensorを避けてdictで対応づける
            id_to_pos = None
            id_to_pos_dict = {
                int(tid): pos
                for pos, tid in enumerate(target_index_cpu.tolist())
            }

            print(
                "[LOG] Using dict target_id-to-position map "
                f"because max_target_id={max_target_id} is large."
            )

        sum_src = torch.zeros((num_targets, node_dim), dtype=torch.float32, device="cpu")
        sum_tgt = torch.zeros((num_targets, node_dim), dtype=torch.float32, device="cpu")
        counts = torch.zeros((num_targets, 1), dtype=torch.float32, device="cpu")

        for start in range(0, total_edges, chunk_size):
            end = min(start + chunk_size, total_edges)

            tgt_ids_chunk = target_edge_index_cpu[1, start:end]  
            attr_chunk = target_edge_attr_cpu[start:end]

            if use_dense_map:
                in_range = (tgt_ids_chunk >= 0) & (tgt_ids_chunk <= max_target_id)

                mapped_pos = torch.full(
                    tgt_ids_chunk.shape,
                    -1,
                    dtype=torch.long,
                    device="cpu"
                )

                mapped_pos[in_range] = id_to_pos[tgt_ids_chunk[in_range]]

            else:
                mapped_pos_list = [
                    id_to_pos_dict.get(int(tid), -1)
                    for tid in tgt_ids_chunk.tolist()
                ]

                mapped_pos = torch.tensor(
                    mapped_pos_list,
                    dtype=torch.long,
                    device="cpu"
                )

            valid = mapped_pos >= 0

            if valid.sum().item() == 0:
                continue

            target_pos = mapped_pos[valid]

            src_feat_valid = attr_chunk[valid, :node_dim]
            tgt_feat_valid = attr_chunk[valid, node_dim:]

            sum_src.index_add_(0, target_pos, src_feat_valid)
            sum_tgt.index_add_(0, target_pos, tgt_feat_valid)

            one_counts = torch.ones((target_pos.numel(), 1), dtype=torch.float32, device="cpu")
            counts.index_add_(0, target_pos, one_counts)

            if start == 0 or end == total_edges or (start // chunk_size) % 20 == 0:
                print(f"[LOG] fixed_edge_diff CPU aggregation: {end}/{total_edges}")

            del tgt_ids_chunk
            del attr_chunk
            del mapped_pos
            del valid
            del target_pos

        counts = counts.clamp_min(1.0)

        mean_src = sum_src / counts
        mean_tgt = sum_tgt / counts

        edge_diff_cpu = mean_src - mean_tgt  # (num_targets, 768)

        with torch.no_grad():
            edge_diff = edge_diff_cpu.to(output_device)
            edge_diff = self.edge_fc(edge_diff)
            edge_diff = (edge_diff - edge_diff.mean(dim=0)) / (edge_diff.std(dim=0) + 1e-6)
            edge_diff = F.relu(edge_diff)

        print(f"[LOG] fixed_edge_diff computed: {tuple(edge_diff.shape)}")
        print(f"[LOG] fixed_edge_diff_target_index shape: {tuple(fixed_edge_diff_target_index_cpu.shape)}")

        if edge_diff.shape[0] != fixed_edge_diff_target_index_cpu.numel():
            raise RuntimeError(
                f"fixed_edge_diff and fixed_edge_diff_target_index length mismatch: "
                f"{edge_diff.shape[0]} vs {fixed_edge_diff_target_index_cpu.numel()}"
            )

        if not torch.equal(fixed_edge_diff_target_index_cpu, target_index.detach().cpu().long()):
            raise RuntimeError(
                "fixed_edge_diff_target_index order was changed unexpectedly. "
                "compute_fixed_edge_diff_cpu must preserve the input target_index order."
            )

        del sum_src
        del sum_tgt
        del counts
        del mean_src
        del mean_tgt
        del edge_diff_cpu

        if id_to_pos is not None:
            del id_to_pos

        if id_to_pos_dict is not None:
            del id_to_pos_dict

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # fixed_edge_diffの各行は fixed_edge_diff_target_index_cpu の同じ行に対応する
        return edge_diff.detach(), fixed_edge_diff_target_index_cpu.detach()

    def forward(self, x, target_edge_index, target_index, target_x_new, fixed_edge_diff):
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

        edge_diff = fixed_edge_diff

        memory_applied = x_diff + x_j

        return memory_applied, x_diff, edge_diff

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

    def compute_fixed_domination_true(
        self,
        target_index,
        target_edge_index,
        target_edge_attr,
        model_target_edge_index,
        target_node_features
    ):
        with torch.no_grad():
            dom_Wj_list = []
            Aj_mean_list = []

            for Aj in target_index:
                mask_Aj = target_edge_index[1] == Aj
                selected_edges = target_edge_index[:, mask_Aj]
                selected_edge_attr = target_edge_attr[mask_Aj]

                if selected_edges.shape[1] == 0:
                    Wj = torch.ones((1,), device=target_edge_attr.device) * 1e-6

                    mask_Aj_in_features = model_target_edge_index[1] == Aj
                    Aj_features = target_node_features[mask_Aj_in_features]

                    if Aj_features.shape[0] > 1:
                        Aj_mean = Aj_features.mean(dim=0, keepdim=True)
                    elif Aj_features.shape[0] == 1:
                        Aj_mean = Aj_features.unsqueeze(0) if Aj_features.dim() == 1 else Aj_features
                    else:
                        Aj_mean = torch.zeros((1, 768), device=target_edge_attr.device)

                    dom_Wj_list.append(Wj.unsqueeze(0))
                    Aj_mean_list.append(Aj_mean)
                    continue

                Ai_ids = selected_edges[0]

                Ai_j_ave_list = []
                Aj_i_ave_list = []

                for Ai in Ai_ids:
                    mask_Ai = selected_edges[0] == Ai
                    edge_attr_AiAj = selected_edge_attr[mask_Ai]

                    if edge_attr_AiAj.shape[0] > 1:
                        Ai_j_ave = edge_attr_AiAj[:, :768].mean(dim=0, keepdim=True)
                        Aj_i_ave = edge_attr_AiAj[:, 768:].mean(dim=0, keepdim=True)
                    else:
                        Ai_j_ave = edge_attr_AiAj[:, :768] if edge_attr_AiAj.ndim > 1 else edge_attr_AiAj[:768].unsqueeze(0)
                        Aj_i_ave = edge_attr_AiAj[:, 768:] if edge_attr_AiAj.ndim > 1 else edge_attr_AiAj[768:].unsqueeze(0)

                    Ai_j_ave_list.append(Ai_j_ave)
                    Aj_i_ave_list.append(Aj_i_ave)

                if len(Ai_j_ave_list) > 1 and len(Aj_i_ave_list) > 1:
                    Ai_j_ave = torch.cat(Ai_j_ave_list, dim=0)
                    Aj_i_ave = torch.cat(Aj_i_ave_list, dim=0)
                elif len(Ai_j_ave_list) == 1 and len(Aj_i_ave_list) == 1:
                    Ai_j_ave = Ai_j_ave_list[0]
                    Aj_i_ave = Aj_i_ave_list[0]
                else:
                    Wj = torch.ones((1,), device=target_edge_attr.device) * 1e-6

                    mask_Aj_in_features = model_target_edge_index[1] == Aj
                    Aj_features = target_node_features[mask_Aj_in_features]

                    if Aj_features.shape[0] > 1:
                        Aj_mean = Aj_features.mean(dim=0, keepdim=True)
                    else:
                        Aj_mean = Aj_features.unsqueeze(0) if Aj_features.dim() == 1 else Aj_features

                    dom_Wj_list.append(Wj.unsqueeze(0))
                    Aj_mean_list.append(Aj_mean)
                    continue

                mask_Aj_in_features = model_target_edge_index[1] == Aj
                Aj_features = target_node_features[mask_Aj_in_features]

                if Aj_features.shape[0] > 1:
                    Aj_mean = Aj_features.mean(dim=0, keepdim=True)
                else:
                    Aj_mean = Aj_features.unsqueeze(0) if Aj_features.dim() == 1 else Aj_features

                Ai_features_list = []
                for Ai in Ai_ids:
                    mask_Ai = model_target_edge_index[1] == Ai
                    Ai_feature = target_node_features[mask_Ai]

                    if Ai_feature.shape[0] > 1:
                        Ai_feature = Ai_feature.mean(dim=0, keepdim=True)
                    else:
                        Ai_feature = Ai_feature.unsqueeze(0) if Ai_feature.dim() == 1 else Ai_feature

                    Ai_features_list.append(Ai_feature)

                if len(Ai_features_list) > 0:
                    Ai_mean = torch.cat(Ai_features_list, dim=0)
                else:
                    Ai_mean = torch.empty((0, 768), device=target_node_features.device)

                if Ai_j_ave.numel() == 0 or Aj_i_ave.numel() == 0:
                    dom_Wj = torch.ones((1,), device=target_edge_attr.device) * 1e-6
                else:
                    dist_Ai = torch.norm(Ai_mean - Ai_j_ave, dim=1)
                    dist_Aj = torch.norm(Aj_mean - Aj_i_ave, dim=1)

                    beta = 0.5
                    d_0 = 1

                    Wi = 1 / (1 + torch.exp(-beta * (dist_Ai - d_0)))
                    Wj = 1 / (1 + torch.exp(-beta * (dist_Aj - d_0)))

                    Wj = Wj.unsqueeze(0) if Wj.dim() == 0 else Wj
                    dom_Wj = Wj.mean(dim=0, keepdim=True)

                dom_Wj_list.append(dom_Wj.unsqueeze(0) if dom_Wj.dim() == 1 else dom_Wj)
                Aj_mean_list.append(Aj_mean.unsqueeze(0) if Aj_mean.dim() == 1 else Aj_mean)

            dom_Wj = torch.cat(dom_Wj_list, dim=0)
            Aj_mean = torch.cat(Aj_mean_list, dim=0)

            fc_device = self.fc_dom_true.weight.device
            domination_input = (dom_Wj * Aj_mean).to(fc_device)

            domination_true = self.fc_dom_true(domination_input)
            domination_true = (domination_true - domination_true.mean(dim=0)) / (domination_true.std(dim=0) + 1e-6)
            domination_true = F.relu(domination_true)

            return domination_true.detach()

    def forward(self, memory_applied):
        dom_out = self.fc1(memory_applied)
        dom_out = self.bn1(dom_out)
        dom_out = F.relu(dom_out)

        dom_out = self.fc2(dom_out)
        dom_out = self.bn2(dom_out)
        dom_out = F.relu(dom_out)

        return dom_out


def align_fixed_tensor_to_target_index(
    fixed_tensor,
    fixed_target_index,
    current_target_index,
    tensor_name="fixed_tensor"
):
    fixed_device = fixed_tensor.device
    current_device = current_target_index.device

    fixed_target_index_dev = fixed_target_index.to(current_device).long()
    current_target_index = current_target_index.long()

    sort_order = torch.argsort(fixed_target_index_dev)
    fixed_target_index_sorted = fixed_target_index_dev[sort_order]

    sort_order_for_tensor = sort_order.to(fixed_device)
    fixed_tensor_sorted = fixed_tensor.index_select(0, sort_order_for_tensor)

    pos = torch.searchsorted(fixed_target_index_sorted, current_target_index)
    safe_pos = pos.clamp(max=fixed_target_index_sorted.numel() - 1)

    matched = fixed_target_index_sorted[safe_pos] == current_target_index
    valid = (pos < fixed_target_index_sorted.numel()) & matched

    if not torch.all(valid):
        missing_ids = current_target_index[~valid].detach().cpu().tolist()
        raise RuntimeError(
            f"{tensor_name}: some current target IDs are missing from fixed_target_index. "
            f"Missing examples: {missing_ids[:20]}"
        )

    aligned_tensor = fixed_tensor_sorted.index_select(
        0,
        safe_pos.to(fixed_device)
    )

    if aligned_tensor.shape[0] != current_target_index.numel():
        raise RuntimeError(
            f"{tensor_name}: aligned row count mismatch. "
            f"aligned={aligned_tensor.shape[0]}, current_target_index={current_target_index.numel()}"
        )

    return aligned_tensor


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
        target_edge_index,
        fixed_edge_diff,
        fixed_domination_true,
        fixed_teacher_target_index
    ):
        target_x_new, target_index, attn_score = self.model_target_gat(
            model_x,
            target_x,
            model_target_edge_index,
            model_target_edge_attr
        )

        fixed_edge_diff_aligned = align_fixed_tensor_to_target_index(
            fixed_tensor=fixed_edge_diff,
            fixed_target_index=fixed_teacher_target_index,
            current_target_index=target_index,
            tensor_name="fixed_edge_diff"
        )

        fixed_domination_true_aligned = align_fixed_tensor_to_target_index(
            fixed_tensor=fixed_domination_true,
            fixed_target_index=fixed_teacher_target_index,
            current_target_index=target_index,
            tensor_name="fixed_domination_true"
        )

        target_x_new = self.feature_enhancement_target(target_x_new)

        target_x_new = self.target_feature_enhancement(target_x_new)

        memory_applied, x_diff, edge_diff = self.target_domination(
            target_x_new,
            target_edge_index,
            target_index,
            target_x_new,
            fixed_edge_diff_aligned
        )

        dom_out = self.domination_layer(memory_applied)

        domination_true = fixed_domination_true_aligned

        if x_diff.shape != edge_diff.shape:
            raise RuntimeError(
                f"x_diff and fixed_edge_diff_aligned shape mismatch: "
                f"x_diff={tuple(x_diff.shape)}, "
                f"edge_diff={tuple(edge_diff.shape)}"
            )

        if dom_out.shape != domination_true.shape:
            raise RuntimeError(
                f"dom_out and fixed_domination_true_aligned shape mismatch: "
                f"dom_out={tuple(dom_out.shape)}, "
                f"domination_true={tuple(domination_true.shape)}"
            )

        return attn_score, memory_applied, x_diff, edge_diff, dom_out, domination_true

total_epochs = 200
gnn_model = GNNModel(
    model_in_channels,
    target_in_channels,
    edge_in_channels,
    target_edge_in_channels,
    total_epochs
).to(device)

optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)

scaler = torch.amp.GradScaler('cuda')


with torch.no_grad():
    fixed_target_index = model_target_edge_index[1].unique()

    fixed_target_index_cpu = fixed_target_index.detach().cpu().long()
    model_target_edge_index_cpu = model_target_edge_index.detach().cpu()
    target_node_features_cpu = target_node_features.detach().cpu()

    fixed_edge_diff, fixed_edge_diff_target_index_cpu = gnn_model.target_domination.compute_fixed_edge_diff_cpu(
        target_index=fixed_target_index_cpu,
        target_edge_index_cpu=target_edge_index_cpu,
        target_edge_attr_cpu=target_edge_attr_cpu,
        node_dim=768,
        chunk_size=200_000,
        output_device=device
    )

    print("pre-calculation of fixed_edge_diff has been completed.")
    print(f"fixed_edge_diff shape: {tuple(fixed_edge_diff.shape)}")
    print(f"fixed_edge_diff_target_index shape: {tuple(fixed_edge_diff_target_index_cpu.shape)}")

    if not torch.equal(fixed_edge_diff_target_index_cpu, fixed_target_index_cpu):
        raise RuntimeError(
            "fixed_edge_diff_target_index_cpu does not match fixed_target_index_cpu. "
            "compute_fixed_edge_diff_cpu must preserve target_index order."
        )

    print("fixed_edge_diff_target_index preserves fixed_target_index order.")
    
    fixed_domination_true_cpu = gnn_model.domination_layer.compute_fixed_domination_true(
        target_index=fixed_edge_diff_target_index_cpu,
        target_edge_index=target_edge_index_cpu,
        target_edge_attr=target_edge_attr_cpu,
        model_target_edge_index=model_target_edge_index_cpu,
        target_node_features=target_node_features_cpu
    ).detach().cpu()

    fixed_domination_true = fixed_domination_true_cpu.to(device)

    fixed_teacher_target_index = fixed_edge_diff_target_index_cpu.to(device)

    print("pre-calculation of fixed_domination_true has been completed.")
    print(f"fixed_domination_true shape: {tuple(fixed_domination_true.shape)}")
    print(f"fixed_teacher_target_index shape: {tuple(fixed_teacher_target_index.shape)}")

    if fixed_edge_diff.shape[0] != fixed_teacher_target_index.numel():
        raise RuntimeError(
            f"fixed_edge_diff and fixed_teacher_target_index length mismatch: "
            f"{fixed_edge_diff.shape[0]} vs {fixed_teacher_target_index.numel()}"
        )

    if fixed_domination_true.shape[0] != fixed_teacher_target_index.numel():
        raise RuntimeError(
            f"fixed_domination_true and fixed_teacher_target_index length mismatch: "
            f"{fixed_domination_true.shape[0]} vs {fixed_teacher_target_index.numel()}"
        )

    torch.save(
        fixed_edge_diff.detach().cpu(),
        os.path.join(output_dir, "fixed_edge_diff.pt")
    )

    torch.save(
        fixed_edge_diff_target_index_cpu,
        os.path.join(output_dir, "fixed_edge_diff_target_index.pt")
    )

    torch.save(
        fixed_domination_true.detach().cpu(),
        os.path.join(output_dir, "fixed_domination_true.pt")
    )

    torch.save(
        fixed_teacher_target_index.detach().cpu(),
        os.path.join(output_dir, "fixed_teacher_target_index.pt")
    )

    print("Saved the fixed teacher values ​​and corresponding target index.")

    del target_edge_attr_cpu
    del fixed_domination_true_cpu
    del target_node_features_cpu
    del model_target_edge_index_cpu

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def plot_pca_projection(data, title, save_path):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    pca = PCA(n_components=2)
    projected = pca.fit_transform(data_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(projected[:, 0], projected[:, 1], alpha=0.5)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(title)
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()


def save_epoch_checkpoint(epoch_num, gnn_model, dom_out, attn_score, loss_records):
    epoch_dir = os.path.join(output_dir, f"{epoch_num:03d}epoch")
    os.makedirs(epoch_dir, exist_ok=True)

    torch.save(gnn_model, os.path.join(epoch_dir, "gnn_model_final.pt"))
    torch.save(gnn_model.state_dict(), os.path.join(epoch_dir, "gnn_model_final_state_dict.pt"))
    torch.save(dom_out, os.path.join(epoch_dir, "final_domination_output.pt"))
    torch.save(attn_score, os.path.join(epoch_dir, "final_attn_score.pt"))

    checkpoint_df = pd.DataFrame(
        loss_records,
        columns=[
            "Epoch", "Edge Loss", "Domination Loss", "Total Loss",
            "Attn Score Mean", "Attn Score Std", "Dom Out Mean", "Dom Out Std"
        ]
    )
    checkpoint_df.to_csv(os.path.join(epoch_dir, "loss_and_attn_score.csv"), index=False)

    print(f"Epoch {epoch_num} checkpoint saved to: {epoch_dir}")


for epoch in range(total_epochs):
    gnn_model.train()  
    optimizer.zero_grad(set_to_none=True)
    
    with torch.amp.autocast('cuda'):  
        attn_score, memory_applied, x_diff, edge_diff, dom_out, domination_true = gnn_model(
            model_node_features, target_node_features,  
            model_target_edge_index, model_target_edge_attr,
            target_edge_index, fixed_edge_diff, fixed_domination_true, fixed_teacher_target_index
        )

        edge_loss = F.mse_loss(x_diff.float(), edge_diff.float())  
        domination_loss = F.mse_loss(dom_out.float(), domination_true.float())  

        loss = 2.0 * edge_loss + 0.1 * domination_loss

    scaler.scale(loss).backward()  
    scaler.unscale_(optimizer)  
    scaler.step(optimizer) 
    scaler.update() 

    attn_mean_per_node = attn_score.mean(dim=1)  

    attn_mean = attn_mean_per_node.mean().item()  
    attn_std = attn_mean_per_node.std().item()  
    
    dom_out_nodewise_mean = dom_out.mean(dim=1)  

    dom_out_mean = dom_out_nodewise_mean.mean().item()  
    dom_out_std = dom_out_nodewise_mean.std().item()  

    loss_records.append([
        epoch + 1, edge_loss.item(), domination_loss.item(), loss.item(),
        attn_mean, attn_std, dom_out_mean, dom_out_std
    ])
    
    print(f"Epoch {epoch+1}, edge_loss: {edge_loss.item()}, domination_loss: {domination_loss.item()}, total_loss: {loss.item()}")


    epoch_num = epoch + 1

    if epoch_num % 50 == 0:
        epoch_dir = os.path.join(output_dir, f"{epoch_num:03d}epoch")
        os.makedirs(epoch_dir, exist_ok=True)

        save_epoch_checkpoint(
            epoch_num=epoch_num,
            gnn_model=gnn_model,
            dom_out=dom_out,
            attn_score=attn_score,
            loss_records=loss_records
        )

        torch.save(dom_out, os.path.join(epoch_dir, f"domination_output_epoch{epoch_num}.pt"))

        attn_score_data = attn_score.detach().cpu().numpy()
        plot_pca_projection(
            attn_score_data,
            f"PCA Projection of Attention Score (Epoch {epoch_num})",
            os.path.join(epoch_dir, f"pca_attn_score_epoch{epoch_num}.png")
        )

        dom_out_data = dom_out.detach().cpu().numpy()
        plot_pca_projection(
            dom_out_data,
            f"PCA Projection of Domination Output (Epoch {epoch_num})",
            os.path.join(epoch_dir, f"pca_dom_out_epoch{epoch_num}.png")
        )

        del attn_score_data
        del dom_out_data

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


gnn_model.eval()

torch.save(gnn_model, os.path.join(output_dir, "gnn_model_final.pt"))  
torch.save(gnn_model.state_dict(), os.path.join(output_dir, "gnn_model_final_state_dict.pt"))  
torch.save(dom_out, os.path.join(output_dir, "final_domination_output.pt"))
torch.save(attn_score, os.path.join(output_dir, "final_attn_score.pt"))

csv_path = os.path.join(output_dir, "loss_and_attn_score.csv")
df = pd.DataFrame(
    loss_records,
    columns=[
        "Epoch", "Edge Loss", "Domination Loss", "Total Loss",
        "Attn Score Mean", "Attn Score Std", "Dom Out Mean", "Dom Out Std"
    ]
)
df.to_csv(csv_path, index=False)

print("model and data have been saved to ./data/gnn_data/")

attn_score_data = attn_score.detach().cpu().numpy()
plot_pca_projection(attn_score_data, "Final PCA Projection of Attention Score", os.path.join(output_dir, "pca_attn_score_final.png"))

dom_out_data = dom_out.detach().cpu().numpy()
plot_pca_projection(dom_out_data, "Final PCA Projection of Domination Output", os.path.join(output_dir, "pca_dom_out_final.png"))

domination_true_data = domination_true.detach().cpu().numpy()  
plot_pca_projection(domination_true_data, "PCA Projection of Domination True", os.path.join(output_dir, "pca_domination_true.png"))

edge_weight_tensor = torch.tensor(edge_weight_list, dtype=torch.float32) 
edge_weight_index_tensor = torch.tensor(edge_weight_index_list, dtype=torch.int64) 

torch.save(edge_weight_tensor, os.path.join(output_dir, "edge_weight.pt"))
torch.save(edge_weight_index_tensor, os.path.join(output_dir, "edge_weight_index.pt"))

print(f"edge_weight.pt and edge_weight_index.pt have been saved to {output_dir}")
