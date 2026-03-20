#!/bin/bash

echo "Running run_all.sh"
all_start_time=$(date +%s)

# Data preparation: prepare data for all datasets (ciphertext)
bash run_data_preparation.sh

# Plaintext training: generate plaintext training results and network
bash run_training_plaintext.sh

# Secure inference performance: run secure inference performance experiments for all datasets
bash secure_inference_performance.sh

# Ablation study for SAI: compare with and without SAI
bash run_SAI.sh

# Ablation study for Batch Array Access: compare with and without batch array access
bash run_BatchAA.sh

# Ablation study for position swap: compare with and without position swap
bash run_PositionSwap.sh

# Ablation study for neighbor padding: compare different neighbor padding modes
bash run_Padding.sh

# Scalability experiments: run scalability experiments for synthetic datasets
bash run_training_synthetic.sh
bash run_scalability_all.sh

all_end_time=$(date +%s)
echo "All experiments completed"