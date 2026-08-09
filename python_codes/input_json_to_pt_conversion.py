import os
import re
import glob
import json
import torch
import gc


DATA_DIR = "./data/input_data/input_temp_json"   
OUTPUT_DIR = "./data/input_data/input_pt_data"   

os.makedirs(OUTPUT_DIR, exist_ok=True)


REQUIRED_JSON_FILES = {
    "edge_index": "input_edge_index.json",
    "edge_attr": "input_edge_attr.json",
    "target_edge_index": "input_target_edge_index.json",
    "target_edge_attr": "input_target_edge_attr.json",
}

OUTPUT_PT_FILES = {
    "edge_index": "input_edge_index.pt",
    "edge_attr": "input_edge_attr.pt",
    "target_edge_index": "input_target_edge_index.pt",
    "target_edge_attr": "input_target_edge_attr.pt",
}


def load_json(file_path):
    if not os.path.exists(file_path):
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_sample_dirs(data_dir):
    dirs = glob.glob(os.path.join(data_dir, "male_*"))

    def extract_number(path):
        name = os.path.basename(os.path.normpath(path))

        m = re.match(r"male_(\d+)$", name)
        if m:
            return int(m.group(1))

        return 10**9

    dirs = [
        d for d in dirs
        if os.path.isdir(d)
        and re.match(r"male_\d+$", os.path.basename(os.path.normpath(d)))
    ]

    dirs = sorted(dirs, key=extract_number)

    return dirs


def validate_required_files(sample_dir):
    missing = []

    for file_name in REQUIRED_JSON_FILES.values():
        path = os.path.join(sample_dir, file_name)
        if not os.path.exists(path):
            missing.append(file_name)

    return missing


def convert_index_json_to_tensor(index_data, tensor_name):
    if index_data is None:
        raise ValueError(f"{tensor_name}: JSON data is None")

    if len(index_data) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    tensor = torch.tensor(index_data, dtype=torch.long)

    if tensor.ndim != 2 or tensor.shape[1] != 2:
        raise ValueError(
            f"{tensor_name}: expected shape (num_edges, 2), "
            f"but got {tuple(tensor.shape)}"
        )

    return tensor.T.contiguous()


def convert_attr_json_to_tensor(attr_data, tensor_name, expected_dim=1536):
    if attr_data is None:
        raise ValueError(f"{tensor_name}: JSON data is None")

    if len(attr_data) == 0:
        return torch.empty((0, expected_dim), dtype=torch.float32)

    tensor = torch.tensor(attr_data, dtype=torch.float32)

    if tensor.ndim != 2:
        raise ValueError(
            f"{tensor_name}: expected 2D tensor, "
            f"but got {tuple(tensor.shape)}"
        )

    if tensor.shape[1] != expected_dim:
        raise ValueError(
            f"{tensor_name}: expected feature dimension {expected_dim}, "
            f"but got {tensor.shape[1]}"
        )

    return tensor.contiguous()


def save_tensor(tensor, output_path):
    """
    Tensorを.ptとして保存する。
    """
    torch.save(tensor, output_path)
    print(f"Saved: {output_path} | shape={tuple(tensor.shape)}")


def convert_one_sample(sample_dir):
    sample_name = os.path.basename(os.path.normpath(sample_dir))
    sample_output_dir = os.path.join(OUTPUT_DIR, sample_name)
    os.makedirs(sample_output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"LOG: Converting sample: {sample_name}")
    print(f"LOG: sample_dir = {sample_dir}")
    print(f"LOG: sample_output_dir = {sample_output_dir}")
    print("=" * 80)

    missing = validate_required_files(sample_dir)

    if missing:
        print(f"Skipping {sample_name}: missing files = {missing}")
        return {
            "sample_name": sample_name,
            "status": "skipped",
            "reason": f"missing files: {missing}",
        }

    edge_index_file = os.path.join(sample_dir, REQUIRED_JSON_FILES["edge_index"])
    edge_attr_file = os.path.join(sample_dir, REQUIRED_JSON_FILES["edge_attr"])
    target_edge_index_file = os.path.join(sample_dir, REQUIRED_JSON_FILES["target_edge_index"])
    target_edge_attr_file = os.path.join(sample_dir, REQUIRED_JSON_FILES["target_edge_attr"])

    edge_index_pt = os.path.join(sample_output_dir, OUTPUT_PT_FILES["edge_index"])
    edge_attr_pt = os.path.join(sample_output_dir, OUTPUT_PT_FILES["edge_attr"])
    target_edge_index_pt = os.path.join(sample_output_dir, OUTPUT_PT_FILES["target_edge_index"])
    target_edge_attr_pt = os.path.join(sample_output_dir, OUTPUT_PT_FILES["target_edge_attr"])

    edge_index = load_json(edge_index_file)
    edge_index_tensor = convert_index_json_to_tensor(
        edge_index,
        tensor_name=f"{sample_name}/input_edge_index"
    )
    save_tensor(edge_index_tensor, edge_index_pt)

    edge_attr = load_json(edge_attr_file)
    edge_attr_tensor = convert_attr_json_to_tensor(
        edge_attr,
        tensor_name=f"{sample_name}/input_edge_attr",
        expected_dim=1536
    )
    save_tensor(edge_attr_tensor, edge_attr_pt)

    if edge_index_tensor.shape[1] != edge_attr_tensor.shape[0]:
        raise RuntimeError(
            f"{sample_name}: input_edge_index and input_edge_attr count mismatch: "
            f"edge_index={edge_index_tensor.shape[1]}, "
            f"edge_attr={edge_attr_tensor.shape[0]}"
        )

    target_edge_index = load_json(target_edge_index_file)
    target_edge_index_tensor = convert_index_json_to_tensor(
        target_edge_index,
        tensor_name=f"{sample_name}/input_target_edge_index"
    )
    save_tensor(target_edge_index_tensor, target_edge_index_pt)

    target_edge_attr = load_json(target_edge_attr_file)
    target_edge_attr_tensor = convert_attr_json_to_tensor(
        target_edge_attr,
        tensor_name=f"{sample_name}/input_target_edge_attr",
        expected_dim=1536
    )
    save_tensor(target_edge_attr_tensor, target_edge_attr_pt)

    if target_edge_index_tensor.shape[1] != target_edge_attr_tensor.shape[0]:
        raise RuntimeError(
            f"{sample_name}: input_target_edge_index and input_target_edge_attr count mismatch: "
            f"target_edge_index={target_edge_index_tensor.shape[1]}, "
            f"target_edge_attr={target_edge_attr_tensor.shape[0]}"
        )

    summary = {
        "sample_name": sample_name,
        "status": "converted",
        "sample_dir": sample_dir,
        "sample_output_dir": sample_output_dir,
        "num_input_edges": int(edge_index_tensor.shape[1]),
        "num_input_edge_attr": int(edge_attr_tensor.shape[0]),
        "num_input_target_edges": int(target_edge_index_tensor.shape[1]),
        "num_input_target_edge_attr": int(target_edge_attr_tensor.shape[0]),
        "output_files": {
            "input_edge_index": edge_index_pt,
            "input_edge_attr": edge_attr_pt,
            "input_target_edge_index": target_edge_index_pt,
            "input_target_edge_attr": target_edge_attr_pt,
        },
    }

    summary_path = os.path.join(sample_output_dir, "pt_conversion_summary.json")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    print(f"📄 Summary saved: {summary_path}")

    del edge_index
    del edge_attr
    del target_edge_index
    del target_edge_attr
    del edge_index_tensor
    del edge_attr_tensor
    del target_edge_index_tensor
    del target_edge_attr_tensor

    gc.collect()

    return summary


sample_dirs = find_sample_dirs(DATA_DIR)

if not sample_dirs:
    raise FileNotFoundError(f"No male_* directories found under {DATA_DIR}")

print(f"LOG: Found {len(sample_dirs)} sample directories:")
for d in sample_dirs:
    print(f"  - {d}")

all_summaries = []

for sample_dir in sample_dirs:
    summary = convert_one_sample(sample_dir)
    all_summaries.append(summary)

all_summary_path = os.path.join(OUTPUT_DIR, "pt_conversion_all_summary.json")

with open(all_summary_path, "w", encoding="utf-8") as f:
    json.dump(all_summaries, f, ensure_ascii=False, indent=4)

print("\nAll input JSON files converted to .pt format successfully.")
print(f"All summary saved: {all_summary_path}")
