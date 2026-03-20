#!/bin/bash

set -e

echo "Running run_Padding.sh (Neighbor Padding ablation)"
all_start_time=$(date +%s)

DATASET=ablation
RUNS=5
NETWORK=gcn
RUN_TYPES=(wan)

for RUN_TYPE in "${RUN_TYPES[@]}"; do
    echo "[${RUN_TYPE}] [Ablation] Padding=random"
    python scripts/run_mpc.py \
        --run_type "$RUN_TYPE" -m test -n "$NETWORK" -d "$DATASET" -r "$RUNS" \
        --sai --position-swap --batch-array-access --base_array_size 2 --mask-neighbor \
        --neighbor_padding_mode random \
        --output "output_ablation/padding_random_${DATASET}"

    echo "[${RUN_TYPE}] [Ablation] Padding=secgnn"
    python scripts/run_mpc.py \
        --run_type "$RUN_TYPE" -m test -n "$NETWORK" -d "$DATASET" -r "$RUNS" \
        --sai --position-swap --batch-array-access --base_array_size 2 --mask-neighbor \
        --neighbor_padding_mode secgnn \
        --output "output_ablation/padding_secgnn_${DATASET}"
done

all_end_time=$(date +%s)
echo "run_Padding.sh completed"
