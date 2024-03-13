#!/bin/bash

# Define the values for your time_encoder and data arguments
DATASET_VALUES=("tgbl-wiki" "tgbl-mooc" "tgbl-canparl")
TIME_ENCODER_VALUES=("learned_cos" "fixed_gaussian" "scaled_fixed" "scaled_fixed_id" "graph_mixer" "decay_amp_gm")

# Fixed values for other parameters
num_run=5
seed=1
edge_step=1
model="DyRep"
mul=1.0

# Iterate over the combinations of time_encoder and data arguments
for time_encoder in "${TIME_ENCODER_VALUES[@]}"; do
  for data in "${DATASET_VALUES[@]}"; do
    # Set the initial time_step value
    time_step=50
    
    # Check if the dataset is "tgbl-canparl" and adjust time_step accordingly
    if [[ "$data" == "tgbl-canparl" ]]; then
      time_step=86400
    fi

    # Call the Python script with the current combination of arguments
    python examples/linkproppred/tgbl-wiki/total_variation_eval.py \
      --data "$data" \
      --num_run "$num_run" \
      --seed "$seed" \
      --time_encoder "$time_encoder" \
      --time_step "$time_step" \
      --edge_step "$edge_step" \
      --model "$model" \
      --mul "$mul"
  done
done