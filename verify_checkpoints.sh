#!/bin/bash

# List of possible values for each variable
DATASET_VALUES=("wiki" "mooc" "canparl")
RUN_IDX_VALUES=("0" "1" "2" "3" "4")
TIME_ENCODER_VALUES=("learned_cos" "fixed_gaussian" "scaled_fixed" "scaled_fixed_id" "graph_mixer" "decay_amp_gm")
MULTIPLIER_VALUES=("1.0")

# Variable to track whether all files exist
all_files_exist=true

# Loop through all combinations and check file existence
for dataset in "${DATASET_VALUES[@]}"; do
  for run_idx in "${RUN_IDX_VALUES[@]}"; do
    for time_encoder in "${TIME_ENCODER_VALUES[@]}"; do
      for multiplier in "${MULTIPLIER_VALUES[@]}"; do
        filename="examples/linkproppred/tgbl-wiki/saved_models/DyRep_tgbl-${dataset}_1_${run_idx}_${time_encoder}_${multiplier}.pth"
        
        # Check if the file exists
        if [ ! -e "$filename" ]; then
          echo "File not found: $filename"
          all_files_exist=false
        fi
      done
    done
  done
done

# Print message if all files exist
if $all_files_exist; then
  echo "All files exist!"
fi