"""Train a best-response to the Roshambo bot(s)."""
from typing import Optional

import jax
import launchpad as lp
import numpy as np
from absl import app
from ml_collections import config_dict

from marl import utils, worlds
from marl.games import openspiel_proxy
from marl_experiments.roshambo import roshambo_bot
from marl_experiments.roshambo.utils import builders
from marl_experiments.roshambo.utils import utils as rps_utils


def get_config() -> config_dict.ConfigDict:
    """Default configuration for the experiment."""
    # Configs that are referred in multiple locations.
    step_key = "learner_steps"
    frame_key = "learner_frames"
    replay_table_name = "learner"
    sequence_length = 20

    config = config_dict.ConfigDict(
        dict(
            result_dir="/scratch/wellman_root/wellman1/mxsmith/results/roshambo/test_br/",
            seed=42,
            impala=config_dict.ConfigDict(
                dict(
                    timestep_encoder_ctor="marl_experiments.roshambo.networks.MLPTimestepEncoder",
                    timestep_encoder_kwargs=dict(),
                    memory_core_ctor="marl_experiments.roshambo.networks.NoopCore",
                    memory_core_kwargs=dict(),
                    policy_head_ctor="marl_experiments.roshambo.networks.PolicyHead",
                    policy_head_kwargs=dict(),
                    value_head_ctor="marl_experiments.roshambo.networks.ValueHead",
                    value_head_kwargs=dict(),
                    discount=0.99,
                    max_abs_reward=np.inf,
                    baseline_cost=0.25,
                    entropy_cost=0.01,
                )
            ),
            learner=config_dict.ConfigDict(
                dict(
                    step_key=step_key,
                    frame_key=frame_key,
                    replay_table_name=replay_table_name,
                    learning_rate=0.0003,
                    max_gradient_norm=40,
                    batch_size=64,
                )
            ),
            replay=config_dict.ConfigDict(
                dict(
                    replay_table_name=replay_table_name,
                    sequence_length=sequence_length,
                    samples_per_insert=1.0,
                    min_size_to_sample=1_000,
                    max_times_sampled=1,
                    error_buffer=2,
                    replay_max_size=1_000_000,
                )
            ),
            training_arena=config_dict.ConfigDict(
                dict(
                    replay_table_name=replay_table_name,
                    sequence_length=sequence_length,
                    variable_update_period=1_000,
                    step_key=step_key,
                    sequence_period=None,
                )
            ),
            evaluation_frequency=10_000,
        )
    )
    return config


def build_game() -> worlds.Game:
    """Builds an instance of repeated RPS."""
    game = openspiel_proxy.OpenSpielProxy("repeated_game(stage_game=matrix_rps(),num_repetitions=1000)")
    return game


def run(config: Optional[config_dict.ConfigDict] = None):
    """Train a response policy according to the experiment configuration."""
    if config is None:
        config = get_config()
    result_dir = utils.ResultDirectory(config.result_dir, overwrite=True, exist_ok=True)
    random_key = jax.random.PRNGKey(config.seed)

    game = rps_utils.build_game()
    env_spec = game.spec()[0]

    bots = {1: roshambo_bot.RoshamboBot(name="rotatebot")}

    _ = lp.Program(name="train_response")

    graphs = builders.build_impala_graphs(config.impala, env_spec)

    counter = builders.build_counter_node()

    replay = builders.build_reverb_node(env_spec=env_spec, state_and_extra_spec=graphs.state_spec, **config.replay)

    random_key, subkey = jax.random.split(random_key)
    learner = builders.build_learner_node(
        loss_graph=graphs.loss,
        replay=replay,
        counter=counter,
        random_key=subkey,
        result_dir=result_dir.make_subdir("learner"),
        **config.learner,
    )

    random_key, subkey = jax.random.split(random_key)
    _ = builders.build_training_arena_node(
        policy_graph=graphs.policy,
        initial_state_graph=graphs.initial_state,
        replay=replay,
        learner=learner,
        game=game,
        bots=bots,
        counter=counter,
        random_key=subkey,
        result_dir=result_dir.make_subdir("train_arena"),
        **config.training_arena,
    )

    _ = builders.build_evaluation_arena_node(
        policy_graph=graphs.eval_policy,
        initial_state_graph=graphs.initial_state,
        game_ctor=build_game,
        learner=learner,
        bots=bots,
        counter=counter,
        random_key=subkey,
        result_dir=result_dir.make_subdir("eval_arena"),
        evaluation_frequency=config.evaluation_frequency,
    )


def main(_):
    """Enables running the file directly through absl, and also running with a config input."""
    run()


if __name__ == "__main__":
    app.run(main)
