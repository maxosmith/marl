#!/bin/bash

sbatch --job-name=2022_12_02_baseline_br_br ../scripts/launch.sh python br_main.py \
    --config.result_dir /scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/2022_12_02_baseline_br_br/ \
    --config.opponent_snapshot_path /scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/2022_11_29_br_sweep/wid_10/snapshot/20221129-163717/impala/ \
    --config.optimizer_config.learning_rate_steps 10000 \
    --config.optimizer_config.max_norm_steps 10000 \
    --config.num_learner_steps 10000
