#!/bin/bash
set -euo pipefail

echo "Running run_synthetic_mpc.sh"
all_start_time=$(date +%s)
echo "All start time: $all_start_time"

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

RUN_TYPE="${RUN_TYPE:-localhost}"
MODE="${MODE:-test}"
NETWORK="${NETWORK:-gcn}"
RUNS="${RUNS:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-output_ciphertext_inference_synthetic}"

DRY_RUN_FLAG="${DRY_RUN_FLAG:-}"
CLEAN_BUILD_FLAG="${CLEAN_BUILD_FLAG:-}"
SKIP_COMPILE_FLAG="${SKIP_COMPILE_FLAG:-}"

echo "Config: RUN_TYPE=${RUN_TYPE}, MODE=${MODE}, NETWORK=${NETWORK}, RUNS=${RUNS}, OUTPUT_DIR=${OUTPUT_DIR}"

# 1) Automatically discover all scalability experiment YAML files
mapfile -t YAML_FILES < <(find parameters -maxdepth 1 -type f -name 'scalability_*.yaml' | sort)
if [[ ${#YAML_FILES[@]} -eq 0 ]]; then
  echo "[ERROR] parameters/scalability_*.yaml not found"
  echo "Please run the synthetic data training and parameter generation scripts first."
  exit 1
fi

echo "Discovered ${#YAML_FILES[@]} scalability YAML files:"
for yaml_file in "${YAML_FILES[@]}"; do
  echo "  - ${yaml_file}"
done

# 2) Pre-check dataset.path and preload_network_path in YAML files
python - << 'PY'
from pathlib import Path
import sys

try:
    from ruamel.yaml import YAML
except Exception:
    print("[ERROR] ruamel.yaml is required. Please install: pip install ruamel.yaml")
    sys.exit(1)

yaml = YAML(typ='safe')
files = sorted(Path('parameters').glob('scalability_*.yaml'))
missing = []
for p in files:
    cfg = yaml.load(p.read_text(encoding='utf-8')) or {}
    ds = (cfg.get('dataset') or {})
    ds_path = ds.get('path')
    net_path = (ds.get('network') or {}).get('preload_network_path')

    if not ds_path or not Path(ds_path).exists():
        missing.append(f"{p}: missing dataset.path -> {ds_path}")
    if not net_path or not Path(net_path).exists():
        missing.append(f"{p}: missing preload_network_path -> {net_path}")

if missing:
  print("[ERROR] Missing paths detected. Please complete the corresponding synthetic data and model training first:")
    for item in missing:
        print("  -", item)
    sys.exit(1)

print(f"Pre-check passed for {len(files)} YAML files.")
PY

# 3) Run all scalability ciphertext inference experiments
CMD=(
  python scripts/run_mpc_synthetic.py
  --run_type "${RUN_TYPE}"
  --mode "${MODE}"
  --network "${NETWORK}"
  --dataset_prefix "scalability_"
  --runs "${RUNS}"
  --output "${OUTPUT_DIR}"
  --sai
  --position-swap
  --batch-array-access
  --mask-neighbor
  --base_array_size 2
)

if [[ -n "${DRY_RUN_FLAG}" ]]; then
  CMD+=(--dry-run)
fi
if [[ -n "${CLEAN_BUILD_FLAG}" ]]; then
  CMD+=(--clean-build)
fi
if [[ -n "${SKIP_COMPILE_FLAG}" ]]; then
  CMD+=(--skip-compile)
fi

echo "Executing: ${CMD[*]}"
"${CMD[@]}"

end_time=$(date +%s)
echo "Synthetic MPC completed, Duration: $((end_time - all_start_time)) seconds"