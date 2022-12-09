#!/bin/bash

sbatch --job-name=2022_12_01_one_step_transfer ../scripts/launch.sh python br_with_planning_main.py -- \
    --config.result_dir /scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/2022_12_01_one_step_transfer/ \
    --config.opponent_snapshot_path /scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/2022_11_29_br_sweep/wid_10/snapshot/20221129-163717/impala/ \
    --config.world_model.path /scratch/wellman_root/wellman1/mxsmith/results/marl/gathering/one_step_transfer/train_world_model/snapshot/20221130-151502/params/
    --config.optimizer_config.learning_rate_steps 10000 \
    --config.optimizer_config.max_norm_steps 10000
