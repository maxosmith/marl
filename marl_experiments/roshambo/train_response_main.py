"""Train a best-response to the Roshambo bot(s)."""
from typing import Optional

import haiku as hk
import launchpad as lp
import numpy as np
import ujson
from absl import app
from ml_collections import config_dict

from marl import strategy, utils, worlds
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

    config = config_dict.create(
        result_dir="/scratch/wellman_root/wellman1/mxsmith/results/roshambo/test_br/",
        seed=42,
        # opponents=["rotatebot"],
        opponents=roshambo_bot.ROSHAMBO_BOT_NAMES,
        opponent_mixture=None,  # Default: uniform.
        impala=config_dict.create(
            timestep_encoder_ctor="marl_experiments.roshambo.networks.MLPTimestepEncoder",
            timestep_encoder_kwargs=dict(),
            memory_core_ctor="marl_experiments.roshambo.networks.MemoryCore",
            memory_core_kwargs=dict(),
            policy_head_ctor="marl_experiments.roshambo.networks.PolicyHead",
            policy_head_kwargs=dict(),
            value_head_ctor="marl_experiments.roshambo.networks.ValueHead",
            value_head_kwargs=dict(),
            discount=0.99,
            max_abs_reward=np.inf,
            baseline_cost=0.25,
            entropy_cost=0.02,
        ),
        learner=config_dict.create(
            step_key=step_key,
            frame_key=frame_key,
            replay_table_name=replay_table_name,
            learning_rate_init=0.0003,
            learning_rate_end=0.0,
            learning_rate_steps=1_000_000,
            max_gradient_norm=40,
            batch_size=64,
        ),
        replay=config_dict.create(
            replay_table_name=replay_table_name,
            sequence_length=sequence_length,
            samples_per_insert=1.0,
            min_size_to_sample=1,
            max_times_sampled=1,
            error_buffer=100,
            replay_max_size=1_000_000,
        ),
        training_arena=config_dict.create(
            replay_table_name=replay_table_name,
            sequence_length=sequence_length,
            step_key=step_key,
            sequence_period=None,
        ),
        num_train_arenas=1,
        evaluation_frequency=100,
    )
    return config


def build_game() -> worlds.Game:
    """Builds an instance of repeated RPS."""
    game = openspiel_proxy.OpenSpielProxy(
        "repeated_game(stage_game=matrix_rps(),num_repetitions=1000)",
        include_full_state=True,
    )
    return game


def run(config: Optional[config_dict.ConfigDict] = None):
    """Train a response policy according to the experiment configuration."""
    if config is None:
        config = get_config()
    if config.opponent_mixture is None:
        # By default, if a mixture over opponent bots is not specified, we assume
        # that the opponent is playing the uniform mixed strategy.
        config.opponent_mixture = (np.ones_like(config.opponents, dtype=float) / len(config.opponents)).tolist()

    result_dir = utils.ResultDirectory(config.result_dir, overwrite=True, exist_ok=True)
    ujson.dump(config.to_dict(), open(result_dir.file("config.json"), "w"))
    key_sequence = hk.PRNGSequence(config.seed)

    game = rps_utils.build_game()
    env_spec = game.spec()[0]

    bots = {
        1: strategy.Strategy(
            policies=[roshambo_bot.RoshamboBot(name) for name in config.opponents],
            mixture=np.asarray(config.opponent_mixture),
        ),
    }

    program = lp.Program(name="train_response")

    graphs = builders.build_impala_graphs(config.impala, env_spec)

    with program.group("counter"):
        counter = program.add_node(builders.build_counter_node())

    with program.group("replay"):
        replay = program.add_node(
            builders.build_reverb_node(
                env_spec=env_spec,
                state_and_extra_spec=graphs.state_spec,
                **config.replay,
            )
        )

    with program.group("learner"):
        learner = program.add_node(
            builders.build_learner_node(
                loss_graph=graphs.loss,
                replay=replay,
                counter=counter,
                random_key=next(key_sequence),
                result_dir=result_dir.make_subdir(program._current_group),
                **config.learner,
            )
        )

    with program.group("train_arena"):
        for i in range(config.num_train_arenas):
            node_name = f"{program._current_group}_{i}"
            _ = program.add_node(
                builders.build_training_arena_node(
                    policy_graph=graphs.policy,
                    initial_state_graph=graphs.initial_state,
                    replay=replay,
                    learner=learner,
                    game=game,
                    bots=bots,
                    counter=counter,
                    random_key=next(key_sequence),
                    result_dir=result_dir.make_subdir(node_name),
                    **config.training_arena,
                )
            )

    with program.group("eval_arena"):
        _ = program.add_node(
            builders.build_evaluation_arena_node(
                policy_graph=graphs.eval_policy,
                initial_state_graph=graphs.initial_state,
                game_ctor=build_game,
                learner=learner,
                bots=bots,
                counter=counter,
                random_key=next(key_sequence),
                result_dir=result_dir.make_subdir(program._current_group),
                evaluation_frequency=config.evaluation_frequency,
                step_key=config.learner.step_key,
            )
        )

    lp.launch(program, launch_type=lp.LaunchType.LOCAL_MULTI_PROCESSING, terminal="current_terminal")


def main(_):
    """Enables running the file directly through absl, and also running with a config input."""
    run()


if __name__ == "__main__":
    app.run(main)
