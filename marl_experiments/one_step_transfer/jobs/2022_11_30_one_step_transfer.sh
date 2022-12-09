#!/bin/bash

sbatch --job-name=2022_11_30_one_step_transfer ../scripts/launch.sh python br_with_planning_main.py \
    --result_dir /scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/2022_11_30_one_step_transfer/ \
    --overrides "opponent_snapshot_path = /scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/2022_11_29_br_sweep/wid_10/snapshot/20221129-163717/impala/"

# This was manually set in the config, because of a bug with flag overrides with `.` in names.
# --overrides "world_model.path = /scratch/wellman_root/wellman1/mxsmith/results/marl/gathering/one_step_transfer/train_world_model/snapshot/20221130-151502/params/"
