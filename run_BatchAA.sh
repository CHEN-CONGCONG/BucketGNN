#!/bin/bash

set -e

echo "Running run_BatchAA.sh (Batch Array Access ablation)"
all_start_time=$(date +%s)

DATASET=ablation
RUNS=5
NETWORK=gcn
RUN_TYPES=(wan)

for RUN_TYPE in "${RUN_TYPES[@]}"; do
    OUTPUT_NO_BATCH="output_ablation/no_batchaa_${DATASET}"
    # OUTPUT_BATCH="output_ablation/batchaa_${DATASET}"

    echo "[${RUN_TYPE}] [Ablation] BatchAA OFF"
    python scripts/run_mpc.py \
        --run_type "$RUN_TYPE" -m test -n "$NETWORK" -d "$DATASET" -r "$RUNS" \
        --sai --position-swap --no-batch-array-access --neighbor_padding_mode bucket \
        --output "$OUTPUT_NO_BATCH" --skip-compile

    # echo "[${RUN_TYPE}] [Ablation] BatchAA ON"
    # python scripts/run_mpc.py \
    #     --run_type "$RUN_TYPE" -m test -n "$NETWORK" -d "$DATASET" -r "$RUNS" \
    #     --sai --position-swap --batch-array-access --base_array_size 2 --neighbor_padding_mode bucket \
    #     --output "$OUTPUT_BATCH" --skip-compile
done

all_end_time=$(date +%s)
echo "run_BatchAA.sh completed"