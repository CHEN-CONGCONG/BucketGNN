#!/bin/bash

echo "Running secure_inference.sh"
all_start_time=$(date +%s)

OUTPUT_DIR="output_ciphertext_inference"


start_time=$(date +%s)
echo "Running secure inferece ..."

python scripts/run_mpc.py --run_type lan -m test -n gcn -d all7 -r 5 --sai --position-swap --batch-array-access --base_array_size 2 --neighbor_padding_mode bucket --output $OUTPUT_DIR
python scripts/run_mpc.py --run_type wan -m test -n gcn -d all7 -r 5 --sai --position-swap --batch-array-access --base_array_size 2 --neighbor_padding_mode bucket --output $OUTPUT_DIR
end_time=$(date +%s)
echo "Running secure inferece completed, Duration: $((end_time - start_time)) seconds"

all_end_time=$(date +%s)
echo "Running secure_inference.sh completed"