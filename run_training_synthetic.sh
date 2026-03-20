#!/bin/bash

echo "Running run_training_synthetic.sh"
all_start_time=$(date +%s)
echo "All start time: $all_start_time"

cd plaintext/

# 1) Node-scale expansion: fixed mean degree and max degree
python train_synthetic_model.py \
  --experiment node_scale \
  --generator kronecker \
  --num_nodes_list 2000 4000 6000 8000 10000 \
  --mean_degree 12 \
  --max_degree 64 \
  --epochs 500 \
  --run 5 \
  --repeats 1 \
  --name_prefix scalability_node \
  --save_model \
  --save_dataset \
  --save_results

# 2) Density-scale expansion: fixed node count and varying mean degree
python train_synthetic_model.py \
  --experiment density_scale \
  --generator kronecker \
  --fixed_num_nodes 4000 \
  --mean_degree_list 4 8 12 16 20 \
  --max_degree 64 \
  --epochs 500 \
  --run 5 \
  --repeats 1 \
  --name_prefix scalability_density \
  --save_model \
  --save_dataset \
  --save_results

end_time=$(date +%s)
echo "Synthetic training completed, Duration: $((end_time - all_start_time)) seconds"

cd ../
