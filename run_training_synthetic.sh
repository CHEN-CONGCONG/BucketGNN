#!/bin/bash

echo "Running run_training_synthetic.sh"
all_start_time=$(date +%s)
echo "All start time: $all_start_time"

cd plaintext/

# 1) 节点规模扩展：固定平均度和最大度
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

# 2) 图密度扩展：固定节点数，扩展平均度
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
