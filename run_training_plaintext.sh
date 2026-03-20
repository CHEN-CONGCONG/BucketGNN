#!/bin/bash

echo "Running run_training_plaintext.sh"
all_start_time=$(date +%s)
echo "All start time: $all_start_time"

echo "Running plaintext training..."
cd plaintext/
python model.py -d all3 -e 500 -r 5
python model.py -d all4 -e 1500 -r 5 -lr 0.01

# python scripts/train_network
end_time=$(date +%s)
echo "Running plaintext training completed, Duration: $((end_time - all_start_time)) seconds"
cd ../  
