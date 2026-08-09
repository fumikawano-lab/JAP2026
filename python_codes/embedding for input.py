import os
import re
import glob
import json
import torch
import gc
import numpy as np
from transformers import BertTokenizer, BertModel
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"LOG: Using device: {device}")

DATA_DIR = "./data"  
print(f"LOG: DATA_DIR = {DATA_DIR}")

try:
    model_name = "."  
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)  
    print(f"LOG: BERT Model ('{model_name}') and Tokenizer loaded successfully!")
except Exception as e:
    print(f"ERROR: Failed to load BERT model ('{model_name}'). {e}")
    exit(1)

def get_embedding(text):
    if isinstance(text, list):
        text = " ".join(map(str, text))
        
    if not text or text.strip().lower() == "none":
        return torch.zeros(768, device=device)  

    try:
        tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)  
        with torch.no_grad():
            outputs = model(**tokens)
        return outputs.last_hidden_state[:, 0, :].squeeze().to(device)  
    except Exception as e:
        print(f"ERROR: Failed to generate embedding for text: {text[:50]}... {e}")
        return torch.zeros(768, device=device) 

INPUT_JSON_DIR = os.path.join(DATA_DIR, "input_data", "individual_data")
INPUT_PATTERN = "male_*.json"

OUTPUT_ROOT_DIR = os.path.join(DATA_DIR, "input_data", "input_temp_json")
os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)

MODEL_DICT_PATH = os.path.join(DATA_DIR, "temp_json", "model_dict.json")
TARGET_DICT_PATH = os.path.join(DATA_DIR, "temp_json", "target_dict.json")


if not os.path.exists(MODEL_DICT_PATH):
    raise FileNotFoundError(f"model_dict.json not found: {MODEL_DICT_PATH}")

if not os.path.exists(TARGET_DICT_PATH):
    raise FileNotFoundError(f"target_dict.json not found: {TARGET_DICT_PATH}")

with open(MODEL_DICT_PATH, "r", encoding="utf-8") as f:
    model_dict = json.load(f)

with open(TARGET_DICT_PATH, "r", encoding="utf-8") as f:
    target_dict = json.load(f)

print(f"LOG: Loaded model_dict: {len(model_dict)} entries")
print(f"LOG: Loaded target_dict: {len(target_dict)} entries")


def find_input_json_files(input_dir, pattern="new_*.json"):
    files = glob.glob(os.path.join(input_dir, pattern))

    sex_order = {
        "male": 0,
        "female": 1,
    }

    def sort_key(path):
        name = os.path.basename(path)

        m = re.match(r"new_(male|female)_(\d+)\.json$", name)
        if m:
            sex = m.group(1)
            number = int(m.group(2))
            return (sex_order.get(sex, 99), number)

        return (99, 10**9)

    files = [
        f for f in files
        if re.match(r"new_(male|female)_\d+\.json$", os.path.basename(f))
    ]

    files = sorted(files, key=sort_key)

    return files


input_json_files = find_input_json_files(INPUT_JSON_DIR, INPUT_PATTERN)

if not input_json_files:
    raise FileNotFoundError(
        f"No input JSON files found: {os.path.join(INPUT_JSON_DIR, INPUT_PATTERN)}"
    )

print(f"LOG: Found {len(input_json_files)} input files:")
for f in input_json_files:
    print(f"  - {f}")


def clean_text(value):
    if value is None:
        return ""
    return str(value).strip().replace("none", "")


def load_json_as_records(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        return [data]

    raise TypeError(f"Unexpected JSON format in {json_path}: {type(data)}")


def rodrigues_rotation(vector, angle_deg, axis):
    device_local = vector.device
    angle_rad = torch.tensor(np.radians(angle_deg), dtype=torch.float32, device=device_local)

    axis = axis.to(device_local) / torch.norm(axis.to(device_local))
    identity = torch.eye(vector.shape[0], device=device_local)
    axis_outer = torch.ger(axis, axis).to(device_local)

    cos_theta = torch.cos(angle_rad)
    sin_theta = torch.sin(angle_rad)

    rotation_matrix = cos_theta * identity + (1 - cos_theta) * axis_outer + sin_theta * torch.diag(axis)
    rotated_vector = torch.matmul(rotation_matrix, vector)

    return rotated_vector


common_rotation_axis = torch.ones(768)
common_rotation_axis = common_rotation_axis / torch.norm(common_rotation_axis)

species_rotation_params = {
    "human": (94.68, 1.7695),
    "mouse": (184.55, 5.1531),
    "rat": (195.33, 5.1250)
}


def process_one_input_json(input_json_path):
    sample_name = os.path.splitext(os.path.basename(input_json_path))[0]
    sample_output_dir = os.path.join(OUTPUT_ROOT_DIR, sample_name)
    os.makedirs(sample_output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"LOG: Processing {sample_name}")
    print(f"LOG: input_json_path = {input_json_path}")
    print(f"LOG: sample_output_dir = {sample_output_dir}")
    print("=" * 80)

    dataset = load_json_as_records(input_json_path)
    print(f"LOG: Loaded {len(dataset)} record(s) from {input_json_path}")

    input_edge_index = []
    input_edge_attr = []
    input_target_edge_index = []
    input_target_edge_attr = []

    skipped_models = []
    skipped_targets = []
    used_model_ids = defaultdict(int)
    used_target_ids = defaultdict(int)

    for index, data in enumerate(dataset):
        paper_id = data.get("id", f"{sample_name}_record_{index}")

        model_main_text = clean_text(data.get("model_main", ""))

        if not model_main_text:
            print(f"WARNING: Skipping record with empty model_main: {paper_id}")
            skipped_models.append({
                "record": paper_id,
                "reason": "empty model_main",
                "model_main": model_main_text
            })
            continue

        if model_main_text not in model_dict:
            print(f"WARNING: model_main_text not found in model_dict -> {model_main_text}")
            skipped_models.append({
                "record": paper_id,
                "reason": "model_main not found in model_dict",
                "model_main": model_main_text
            })
            continue

        model_id = int(model_dict[model_main_text])
        used_model_ids[model_id] += 1

        model_features_text = " ".join([
            clean_text(data.get("species", "")),
            clean_text(data.get("age", "")),
            clean_text(data.get("sex", "")),
            clean_text(data.get("biosample_main", "")),
            clean_text(data.get("biosample_detail", "")),
            clean_text(data.get("experiment_type", "")),
            clean_text(data.get("model_main", "")),
            clean_text(data.get("model_detail1", "")),
            clean_text(data.get("model_detail2", "")),
            clean_text(data.get("model_detail3", "")),
            clean_text(data.get("timepoint", ""))
        ]).strip()

        model_features_embedding = get_embedding(model_features_text)

        species = clean_text(data.get("species", "")).lower()

        if species in species_rotation_params:
            angle, norm_factor = species_rotation_params[species]
            rotation_axis = common_rotation_axis.to(model_features_embedding.device)

            model_features_embedding = rodrigues_rotation(
                model_features_embedding,
                angle,
                rotation_axis
            )
            model_features_embedding = model_features_embedding * norm_factor

        targets = data.get("targets", [])

        if isinstance(targets, dict):
            targets = [targets]

        if not isinstance(targets, list):
            print(f"ERROR: Unexpected format for targets in {paper_id}: {type(targets)}")
            targets = []

        # このrecord内でtarget-target edgeを作るための一時バッファ
        record_target_ids = []
        record_edges_dict = {}

        record_edge_index = []
        record_edge_attr = []

        for target in targets:
            if not isinstance(target, dict):
                print(f"WARNING: Skipping non-dict target in {paper_id}: {target}")
                continue

            target_text = clean_text(target.get("target", ""))

            if not target_text:
                print(f"WARNING: Skipping target with empty name -> {target}")
                skipped_targets.append({
                    "record": paper_id,
                    "reason": "empty target",
                    "target": target_text
                })
                continue

            if target_text not in target_dict:
                print(f"WARNING: target_text not found in target_dict -> {target_text}")
                skipped_targets.append({
                    "record": paper_id,
                    "reason": "target not found in target_dict",
                    "target": target_text
                })
                continue

            target_id = int(target_dict[target_text])
            used_target_ids[target_id] += 1
            record_target_ids.append(target_id)

            edge_weight_text = " ".join([
                clean_text(target.get("target", "")),
                clean_text(target.get("biosample", "")),
                clean_text(target.get("molecule_type", "")),
                clean_text(target.get("analysis_main", "")),
                clean_text(target.get("analysis_detail", "")),
                clean_text(target.get("relation", "")),
                clean_text(target.get("change", "")),
                clean_text(target.get("significance", "")),
                clean_text(target.get("control", ""))
            ]).strip()

            relation = clean_text(target.get("relation", "")).lower()

            edge_weight_embedding = get_embedding(edge_weight_text)

            if relation == "increase":
                edge_weight_embedding = rodrigues_rotation(
                    edge_weight_embedding,
                    90,
                    common_rotation_axis
                )
            elif relation == "decrease":
                edge_weight_embedding = rodrigues_rotation(
                    edge_weight_embedding,
                    -90,
                    common_rotation_axis
                )

            # Model-Target edge feature
            edge_feature = torch.cat(
                (model_features_embedding, edge_weight_embedding),
                dim=0
            )

            # record内のedge情報を保持
            if (model_id, target_id) not in record_edges_dict:
                record_edges_dict[(model_id, target_id)] = []

            record_edges_dict[(model_id, target_id)].append({
                "model_features": model_features_embedding,
                "edge_weight": edge_weight_embedding
            })

            record_edge_index.append([model_id, target_id])
            record_edge_attr.append(edge_feature)

        record_target_edge_index = []
        record_target_edge_attr = []
        record_target_edges_with_weights = {}

        for i in range(len(record_target_ids)):
            for j in range(i + 1, len(record_target_ids)):
                target_id_i = record_target_ids[i]
                target_id_j = record_target_ids[j]

                if (
                    (target_id_i, target_id_j) in record_target_edges_with_weights
                    or (target_id_j, target_id_i) in record_target_edges_with_weights
                ):
                    continue

                edge_info_i = record_edges_dict.get((model_id, target_id_i), [])
                edge_info_j = record_edges_dict.get((model_id, target_id_j), [])

                if isinstance(edge_info_i, list) and len(edge_info_i) > 0:
                    edge_info_i = edge_info_i[-1]

                if isinstance(edge_info_j, list) and len(edge_info_j) > 0:
                    edge_info_j = edge_info_j[-1]

                if isinstance(edge_info_i, dict) and isinstance(edge_info_j, dict):
                    edge_weights_i = edge_info_i["edge_weight"]
                    edge_weights_j = edge_info_j["edge_weight"]
                else:
                    print(
                        f"WARNING: Skipping target-target edge "
                        f"{target_id_i} <-> {target_id_j} due to missing edge_weight."
                    )
                    continue

                # 双方向edge
                record_target_edges_with_weights[(target_id_i, target_id_j)] = [{
                    "edge_weight_source": edge_weights_i,
                    "edge_weight_target": edge_weights_j
                }]
                record_target_edges_with_weights[(target_id_j, target_id_i)] = [{
                    "edge_weight_source": edge_weights_j,
                    "edge_weight_target": edge_weights_i
                }]

        for (target_id_i, target_id_j), weight_list in record_target_edges_with_weights.items():
            for weight in weight_list:
                record_target_edge_index.append([target_id_i, target_id_j])
                record_target_edge_attr.append(
                    torch.cat(
                        (
                            weight["edge_weight_source"],
                            weight["edge_weight_target"]
                        ),
                        dim=0
                    )
                )

        input_edge_index.extend(record_edge_index)
        input_edge_attr.extend(record_edge_attr)
        input_target_edge_index.extend(record_target_edge_index)
        input_target_edge_attr.extend(record_target_edge_attr)

        print(
            f"LOG: Processed record {index + 1}/{len(dataset)} "
            f"for {sample_name}: "
            f"model-target edges={len(record_edge_index)}, "
            f"target-target edges={len(record_target_edge_index)}"
        )

        torch.cuda.empty_cache()
        gc.collect()

    json_files = {
        "input_edge_index": os.path.join(sample_output_dir, "input_edge_index.json"),
        "input_edge_attr": os.path.join(sample_output_dir, "input_edge_attr.json"),
        "input_target_edge_index": os.path.join(sample_output_dir, "input_target_edge_index.json"),
        "input_target_edge_attr": os.path.join(sample_output_dir, "input_target_edge_attr.json"),
    }

    with open(json_files["input_edge_index"], "w", encoding="utf-8") as f:
        json.dump(input_edge_index, f, indent=4)

    with open(json_files["input_edge_attr"], "w", encoding="utf-8") as f:
        json.dump([t.detach().cpu().tolist() for t in input_edge_attr], f, indent=4)

    with open(json_files["input_target_edge_index"], "w", encoding="utf-8") as f:
        json.dump(input_target_edge_index, f, indent=4)

    with open(json_files["input_target_edge_attr"], "w", encoding="utf-8") as f:
        json.dump([t.detach().cpu().tolist() for t in input_target_edge_attr], f, indent=4)

    summary = {
        "sample_name": sample_name,
        "input_json_path": input_json_path,
        "output_dir": sample_output_dir,
        "num_records": len(dataset),
        "num_input_edge_index": len(input_edge_index),
        "num_input_edge_attr": len(input_edge_attr),
        "num_input_target_edge_index": len(input_target_edge_index),
        "num_input_target_edge_attr": len(input_target_edge_attr),
        "num_used_model_ids": len(used_model_ids),
        "num_used_target_ids": len(used_target_ids),
        "used_model_ids": {str(k): int(v) for k, v in used_model_ids.items()},
        "used_target_ids": {str(k): int(v) for k, v in used_target_ids.items()},
        "num_skipped_models": len(skipped_models),
        "num_skipped_targets": len(skipped_targets),
        "skipped_models": skipped_models,
        "skipped_targets": skipped_targets,
        "dictionary_source": {
            "model_dict": MODEL_DICT_PATH,
            "target_dict": TARGET_DICT_PATH
        },
        "id_policy": "use_existing_model_dict_and_target_dict_only; unregistered entries are skipped"
    }

    with open(os.path.join(sample_output_dir, "processing_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    print(f"📄 Saved input embedding JSON files to: {sample_output_dir}")
    print(f"LOG: {sample_name} summary:")
    print(f"  model-target edges: {len(input_edge_index)}")
    print(f"  target-target edges: {len(input_target_edge_index)}")
    print(f"  skipped models: {len(skipped_models)}")
    print(f"  skipped targets: {len(skipped_targets)}")

    del input_edge_index
    del input_edge_attr
    del input_target_edge_index
    del input_target_edge_attr

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


for input_json_path in input_json_files:
    process_one_input_json(input_json_path)

print("\n All individual input JSON files were processed.")
