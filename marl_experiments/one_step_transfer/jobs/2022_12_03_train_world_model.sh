#!/bin/bash

sbatch --job-name=2022_12_03_train_world_model ../scripts/launch.sh python train_world_model_main.py \
    --config.result_dir /scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/2022_12_03_train_world_model
