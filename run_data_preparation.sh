#!/bin/bash

echo "Running run_data_preparation.sh"

all_start_time=$(date +%s)
echo "All start time: $all_start_time"
cd plaintext/
datasets=("cora" "citeseer" "pubmed" "amazon-computer" "amazon-photo" "coauthor-cs" "coauthor-physics")
for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    python dataset.py --dataset "$dataset"
    echo "Finished processing dataset: $dataset"
done

end_time=$(date +%s)
echo "Data preparation completed, Duration: $((end_time - all_start_time)) seconds"

cd ../
