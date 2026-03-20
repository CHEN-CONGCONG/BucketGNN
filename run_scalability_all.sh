#!/bin/bash
set -euo pipefail

echo "Running run_scalability_all.sh"
all_start_time=$(date +%s)
echo "All start time: $all_start_time"

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

RUN_TYPES=("lan" "wan")
MODE="test"
NETWORK="gcn"
RUNS="5"
OUTPUT_DIR="output_ciphertext_inference_synthetic"
OUTPUT_DIR_PANTHER_NO_BAA="output_ciphertext_inference_synthetic_panther_no_batch_array_access"
export RUNS

NODE_SIZES=(2000 4000 6000 8000 10000)
DENSITY_MEAN_DEGREES=(4 8 12 16 20)
MAX_DEGREE=64

echo "Preparing full scalability YAML list..."

python - << 'PY'
import os
from pathlib import Path
import re

try:
    from ruamel.yaml import YAML
except Exception:
    print("[ERROR] ruamel.yaml is required. Please install: pip install ruamel.yaml")
    raise SystemExit(1)

yaml_rt = YAML()
yaml_rt.preserve_quotes = True

root = Path('.')
params_dir = root / 'parameters'
params_dir.mkdir(parents=True, exist_ok=True)
runs = int(os.environ.get('RUNS', '1'))

node_sizes = [2000, 4000, 6000, 8000, 10000]
density_mean_degrees = [4, 8, 12, 16, 20]

targets = []
for n in node_sizes:
    targets.append(f"scalability_node_kronecker_node_scale_n{n}_md12p0_cap64_r0")
for md in density_mean_degrees:
    targets.append(f"scalability_density_kronecker_density_scale_n4000_md{md}p0_cap64_r0")

def build_yaml_obj(name: str, num_nodes: int, mean_degree: float, num_classes: int, num_features: int, num_train: int, num_val_total: int, num_test: int, max_degree: int, hidden_size: int):
    num_edges = int(round(num_nodes * mean_degree / 2.0))
    return {
        'dataset': {
            'name': name,
            'path': f'data/{name}',
            'sai': True,
            'position_swap': True,
            'properties': {
                'num_classes': int(num_classes),
                'num_features': int(num_features),
                'num_nodes': int(num_nodes),
                'num_edges': int(num_edges),
                'num_train': int(num_train),
                'num_val': int(num_val_total),
                'num_test': int(num_test),
                'max_degree': int(max_degree),
            },
            'network': {
                'no_of_runs': 4,
                'hidden_layer_size': int(hidden_size),
                'preload_network_path': f'data/{name}/network',
                'large_network': False,
                'with_normalization': False,
            },
            'graph': {
                'mask_neighbor': True,
                'neighbor_padding_mode': 'bucket',
                'bucket': {
                    'strategy': 'custom',
                    'merge_mode': 'partial_fill',
                    'min_nodes_k': 20,
                    'count': 10,
                    'exp_base': 2,
                    'boundaries': [4, 8, 16, 32, 64],
                },
                'export_mask_distribution': True,
                'mask_distribution_output_dir': 'output/mask_distribution',
                'save_node_init': False,
                'rewrite_node_init': True,
                'node_init_path': f'data/{name}/node_init',
                'precompute_node_state': False,
                'node_state_path': f'data/{name}/node_state',
                'batch_array_access': True,
                'base_array_size': 2,
                'batch_node_size_array_access': 1000,
            },
        }
    }

for name in targets:
    yaml_path = params_dir / f'{name}.yaml'
    data_dir = root / 'data' / name
    info_path = data_dir / 'dataset_info.txt'

    if not data_dir.exists():
        print(f"[ERROR] Missing dataset directory: {data_dir}")
        print("请先运行: bash run_training_synthetic.sh")
        raise SystemExit(1)

    missing_model_files = []
    for run_idx in range(runs):
        model_dir = data_dir / 'network' / 'gcn' / f'run_{run_idx}'
        required_files = [
            'fc_weight_1_A', 'fc_weight_1_B', 'fc_weight_1_C',
            'fc_bias_1_A', 'fc_bias_1_B', 'fc_bias_1_C',
            'fc_weight_2_A', 'fc_weight_2_B', 'fc_weight_2_C',
            'fc_bias_2_A', 'fc_bias_2_B', 'fc_bias_2_C',
        ]
        for filename in required_files:
            if not (model_dir / filename).exists():
                missing_model_files.append(str(model_dir / filename))

    if missing_model_files:
        print(f"[ERROR] Missing pretrained model files for dataset: {name}")
        preview = missing_model_files[:6]
        for path in preview:
            print(f"  - {path}")
        remaining = len(missing_model_files) - len(preview)
        if remaining > 0:
            print(f"  ... and {remaining} more")
        print("请先运行: bash run_training_synthetic.sh")
        print("注意: 该脚本现在会导出 network/gcn/run_x 下的模型文件。")
        raise SystemExit(1)

    # default values
    num_classes = 7
    num_features = 128
    num_nodes = 4000
    num_train = int(num_nodes * 0.2)
    num_val_total = int(num_nodes * 0.4)
    num_test = num_nodes - num_val_total
    max_degree = 64
    hidden_size = 64
    mean_degree = 12.0

    if '_node_scale_' in name:
        match = re.search(r'_scale_n(\d+)_md', name)
        if not match:
            raise ValueError(f"Failed to parse node scale from dataset name: {name}")
        num_nodes = int(match.group(1))
        mean_degree = 12.0
    else:
        num_nodes = 4000
        match = re.search(r'_md([0-9]+p[0-9]+)_cap', name)
        if not match:
            raise ValueError(f"Failed to parse mean degree from dataset name: {name}")
        md_token = match.group(1).replace('p', '.')
        mean_degree = float(md_token)

    if info_path.exists():
        for line in info_path.read_text(encoding='utf-8').splitlines():
            if line.startswith('Number of classes:'):
                num_classes = int(line.split(':', 1)[1].strip())
            elif line.startswith('Number of node features:'):
                num_features = int(line.split(':', 1)[1].strip())
            elif line.startswith('Number of nodes:'):
                num_nodes = int(line.split(':', 1)[1].strip())
            elif line.startswith('Train index shape:'):
                token = line.split('torch.Size([', 1)[1].split('])', 1)[0]
                num_train = int(token)
            elif line.startswith('Valid index shape:'):
                token = line.split('torch.Size([', 1)[1].split('])', 1)[0]
                num_val_total = int(token)
            elif line.startswith('Test index shape:'):
                token = line.split('torch.Size([', 1)[1].split('])', 1)[0]
                num_test = int(token)
            elif line.startswith('Max degree'):
                max_degree = int(float(line.split(':', 1)[1].strip()))

    obj = build_yaml_obj(
        name=name,
        num_nodes=num_nodes,
        mean_degree=mean_degree,
        num_classes=num_classes,
        num_features=num_features,
        num_train=num_train,
        num_val_total=num_val_total,
        num_test=num_test,
        max_degree=max_degree,
        hidden_size=hidden_size,
    )

    with yaml_path.open('w', encoding='utf-8') as f:
        yaml_rt.dump(obj, f)
    print(f"[OK] generated {yaml_path}")

print(f"Prepared {len(targets)} scalability YAML files.")
PY

echo "Running all node-scale + density-scale MPC tests..."
for RUN_TYPE in "${RUN_TYPES[@]}"; do
    echo "Starting run_type=${RUN_TYPE} [ours: bucket + batch-array-access]"
    python scripts/run_mpc_synthetic.py \
        --run_type "${RUN_TYPE}" \
        --mode "${MODE}" \
        --network "${NETWORK}" \
        --dataset_names \
            scalability_node_kronecker_node_scale_n2000_md12p0_cap64_r0 \
            scalability_node_kronecker_node_scale_n4000_md12p0_cap64_r0 \
            scalability_node_kronecker_node_scale_n6000_md12p0_cap64_r0 \
            scalability_node_kronecker_node_scale_n8000_md12p0_cap64_r0 \
            scalability_node_kronecker_node_scale_n10000_md12p0_cap64_r0 \
            scalability_density_kronecker_density_scale_n4000_md4p0_cap64_r0 \
            scalability_density_kronecker_density_scale_n4000_md8p0_cap64_r0 \
            scalability_density_kronecker_density_scale_n4000_md12p0_cap64_r0 \
            scalability_density_kronecker_density_scale_n4000_md16p0_cap64_r0 \
            scalability_density_kronecker_density_scale_n4000_md20p0_cap64_r0 \
        --runs "${RUNS}" \
        --output "${OUTPUT_DIR}" \
        --sai \
        --position-swap \
        --batch-array-access \
        --mask-neighbor \
        --neighbor_padding_mode bucket \
        --bucket_padding_strategy custom \
        --bucket_merge_mode partial_fill \
        --bucket_min_nodes_k 20 \
        --bucket_boundaries 4,8,16,32,64 \
        --base_array_size 2

    echo "Starting run_type=${RUN_TYPE} [baseline: panther + no batch-array-access]"
    python scripts/run_mpc_synthetic.py \
        --run_type "${RUN_TYPE}" \
        --mode "${MODE}" \
        --network "${NETWORK}" \
        --dataset_names \
            scalability_node_kronecker_node_scale_n2000_md12p0_cap64_r0 \
            scalability_node_kronecker_node_scale_n4000_md12p0_cap64_r0 \
            scalability_node_kronecker_node_scale_n6000_md12p0_cap64_r0 \
            scalability_node_kronecker_node_scale_n8000_md12p0_cap64_r0 \
            scalability_node_kronecker_node_scale_n10000_md12p0_cap64_r0 \
            scalability_density_kronecker_density_scale_n4000_md4p0_cap64_r0 \
            scalability_density_kronecker_density_scale_n4000_md8p0_cap64_r0 \
            scalability_density_kronecker_density_scale_n4000_md12p0_cap64_r0 \
            scalability_density_kronecker_density_scale_n4000_md16p0_cap64_r0 \
            scalability_density_kronecker_density_scale_n4000_md20p0_cap64_r0 \
        --runs "${RUNS}" \
        --output "${OUTPUT_DIR_PANTHER_NO_BAA}" \
        --sai \
        --position-swap \
        --no-batch-array-access \
        --mask-neighbor \
        --neighbor_padding_mode random \
        --base_array_size 2
done

end_time=$(date +%s)
echo "All scalability MPC tests completed, Duration: $((end_time - all_start_time)) seconds"