#!/bin/bash

sbatch --job-name=2022_12_06_train_world_model_memory ../scripts/launch.sh python train_world_model_main.py \
    --config.result_dir /scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/2022_12_06_train_world_model_memory \
    --config.world_model.memory_core_ctor marl_experiments.gathering.networks.MemoryCore

sbatch --job-name=2022_12_06_train_world_model_nomemory ../scripts/launch.sh python train_world_model_main.py \
    --config.result_dir /scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/2022_12_06_train_world_model_nomemory \
    --config.world_model.memory_core_ctor marl_experiments.gathering.networks.MemoryLessCore
