#!/bin/bash

sbatch --job-name=2022_12_12_one_step_transfer ../scripts/launch_gpu.sh python br_with_planning_main.py \
    --result_dir /scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/2022_12_12_one_step_transfer/ \
    --config.opponent_snapshot_path /scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/2022_12_06_br_random_with_memory/wid_10/eval_arena/20221206-095632_59.79999923706055/ \
    --config.world_model_path /scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/2022_12_10_reward_sweep/wid_0/snapshot/20221211-140851/params/ \
    --config.num_planner_steps 10000 \
    --config.num_learner_steps 10000 \
    --config.agent.baseline_cost 0.2 \
    --config.agent.entropy_cost 0.04 \
    --config.agent_optimizer.learning_rate_init 0.000006 \
    --config.agent_optimizer.max_norm_init 10.0
