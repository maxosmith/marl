#!/bin/bash

sbatch --job-name=2022_11_30_baseline_br_br2 ../scripts/launch.sh python br_main.py \
    --result_dir /scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/2022_11_30_baseline_br_br/ \
    --overrides "opponent_snapshot_path = /scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/2022_11_29_br_sweep/wid_10/snapshot/20221129-163717/impala/"
