#!/bin/bash
#
# result_dir="/scratch/wellman_root/wellman1/mxsmith/results/marl/gathering/impala/",
# seed=42,
# discount=0.99,
# sequence_length=20,
# sequence_period=None,
# step_key="learner_steps",
# frame_key="learner_frames",
# # Termination condition.
# max_steps=4_000,  # Roughly one hour of walltime.
# # Topology.
# num_training_arenas=4,
# # Agent configuration.
# timestep_encoder_ctor="marl_experiments.gathering.networks.MLPTimestepEncoder",
# timestep_encoder_kwargs={},
# memory_core_ctor="marl_experiments.gathering.networks.MemoryLessCore",
# memory_core_kwargs={},
# policy_head_ctor="marl_experiments.gathering.networks.PolicyHead",
# policy_head_kwargs={},
# value_head_ctor="marl_experiments.gathering.networks.ValueHead",
# value_head_kwargs={},
# # Optimizer configuration.
# batch_size=128,
# learning_rate_init=6e-4,
# learning_rate_end=6e-8,
# learning_rate_steps=100_000,
# max_gradient_norm=40.0,
# # Loss configuration.
# baseline_cost=0.25,
# entropy_cost=0.02,
# max_abs_reward=np.inf,
# # Replay options.
# replay_table_name=reverb_adders.DEFAULT_PRIORITY_TABLE,
# replay_max_size=1_000_000,
# samples_per_insert=1.0,
# min_size_to_sample=1,
# max_times_sampled=1,
# error_buffer=100,
# num_prefetch_threads=None,
# max_queue_size=500,
# # Evaluation.
# render_frequency=100,


sbatch --job-name=2022_11_16_baseline ./launch.sh ../experiment.py \
    --result_dir /scratch/wellman_root/wellman1/mxsmith/results/marl/gathering/2022_11_16_baseline


sbatch --job-name=2022_11_16_baseline_lre7 ./launch.sh ../experiment.py \
    --result_dir /scratch/wellman_root/wellman1/mxsmith/results/marl/gathering/2022_11_16_baseline_lre7 \
    --overrides "learning_rate_end = 6e-7"
sbatch --job-name=2022_11_16_baseline_lre9 ./launch.sh ../experiment.py \
    --result_dir /scratch/wellman_root/wellman1/mxsmith/results/marl/gathering/2022_11_16_baseline_lre9 \
    --overrides "learning_rate_end = 6e-9"


sbatch --job-name=2022_11_16_baseline_bs256 ./launch.sh ../experiment.py \
    --result_dir /scratch/wellman_root/wellman1/mxsmith/results/marl/gathering/2022_11_16_baseline_bs256 \
    --overrides "batch_size = 256"


sbatch --job-name=2022_11_16_baseline_gn1 ./launch.sh ../experiment.py \
    --result_dir /scratch/wellman_root/wellman1/mxsmith/results/marl/gathering/2022_11_16_baseline_gn1 \
    --overrides "max_gradient_norm = 1.0"
sbatch --job-name=2022_11_16_baseline_gn10 ./launch.sh ../experiment.py \
    --result_dir /scratch/wellman_root/wellman1/mxsmith/results/marl/gathering/2022_11_16_baseline_gn10 \
    --overrides "max_gradient_norm = 10.0"
sbatch --job-name=2022_11_16_baseline_gn01 ./launch.sh ../experiment.py \
    --result_dir /scratch/wellman_root/wellman1/mxsmith/results/marl/gathering/2022_11_16_baseline_gn01 \
    --overrides "max_gradient_norm = 0.1"
