"""Base configurations."""
import numpy as np
from ml_collections import config_dict

from marl.services.replay.reverb import adders as reverb_adders


def agent():
    """IMPALA agent graph."""
    return config_dict.create(
        timestep_encoder_ctor="marl_experiments.gathering.networks.MLPTimestepEncoder",
        timestep_encoder_kwargs={},
        memory_core_ctor="marl_experiments.gathering.networks.MemoryCore",
        memory_core_kwargs={},
        policy_head_ctor="marl_experiments.gathering.networks.PolicyHead",
        policy_head_kwargs={},
        value_head_ctor="marl_experiments.gathering.networks.ValueHead",
        value_head_kwargs={},
        # Loss configuration.
        baseline_cost=0.2,
        entropy_cost=0.04,
        max_abs_reward=np.inf,
        discount=0.99,
    )


def agent_replay():
    """Replay buffer for training an IMPALA agent."""
    return config_dict.create(
        replay_table_name=reverb_adders.DEFAULT_PRIORITY_TABLE,
        replay_max_size=1_000_000,
        samples_per_insert=1.0,
        min_size_to_sample=1,
        max_times_sampled=1,
        error_buffer=100,
        num_prefetch_threads=None,
        max_queue_size=500,
        sequence_length=20,
        sequence_period=None,
    )


def agent_optimizer():
    """Optimizer for training an IMPALA agent."""
    return config_dict.create(
        optimizer_name="adam",
        batch_size=128,
        # Learning rate: linear schedule.
        learning_rate_init=6e-6,
        learning_rate_end=6e-6,
        learning_rate_steps=None,
        # Maximum gradient norm: linear schedule.
        max_norm_init=10.0,
        max_norm_end=10.0,
        max_norm_steps=None,
    )


def world_model():
    """World model graph."""
    return config_dict.create(
        memory_core_ctor="marl_experiments.gathering.networks.MemoryCore",
        memory_core_kwargs={},
        # Loss function.
        reward_cost=10.0,
    )


def world_replay():
    """Replay buffer for a World Model."""
    return config_dict.create(
        replay_table_name=reverb_adders.DEFAULT_PRIORITY_TABLE,
        replay_max_size=1_000_000,
        sequence_length=20,
        sequence_period=None,
    )


def world_optimizer():
    """Optimizier for training a World Model."""
    return config_dict.create(
        batch_size=128,
        learning_rate=3e-4,
        max_gradient_norm=10.0,
        # Reverb prefetch client.
        prefetch_buffer_size=300,
        prefetch_num_threads=16,
    )
