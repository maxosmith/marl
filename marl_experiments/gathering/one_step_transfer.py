"""One-step model transfer experiment.

This experiment has the following steps:
    1. Train a BR against a fixed policy.
    2. Generate a dataset to train a world-model using the frozen BR and fixed policy.
    3. Train a world-model using the dataset.
    4. Warm start a BR(BR) using the frozen world-model.
    5. Fine tune the BR(BR) using the true world.
    6. Optionally, train a baseline BR(BR) without warm starting.
"""
import dataclasses
import functools
from typing import Any, List, Mapping, Optional

import haiku as hk
import launchpad as lp
import numpy as np
import ujson
from absl import app, flags, logging
from ml_collections import config_dict

from marl import bots as bots_lib
from marl import games, services, utils, worlds
from marl.agents import impala
from marl.services.replay.reverb import adders as reverb_adders
from marl.utils import flag_utils, import_utils, signals, spec_utils, wrappers
from marl_experiments.gathering import build_services, networks
from marl_experiments.gathering import world_model as world_model_lib
from marl_experiments.gathering.one_step_transfer import builders, graphs

flag_utils.common_experiment_flags()
flags.adopt_module_key_flags(flag_utils)
FLAGS = flags.FLAGS

_EPISODE_LENGTH = 100


def get_config() -> config_dict.ConfigDict:
    """Default configuration for this experiment."""
    step_key = "learner_step"
    frame_key = "learner_frame"

    num_steps = 10

    config = config_dict.create(
        result_dir="/scratch/wellman_root/wellman1/mxsmith/results/marl/gathering/one_step_transfer/",
        seed=42,
        agent=config_dict.create(
            timestep_encoder_ctor="marl_experiments.gathering.networks.MLPTimestepEncoder",
            timestep_encoder_kwargs={},
            memory_core_ctor="marl_experiments.gathering.networks.MemoryLessCore",
            memory_core_kwargs={},
            policy_head_ctor="marl_experiments.gathering.networks.PolicyHead",
            policy_head_kwargs={},
            value_head_ctor="marl_experiments.gathering.networks.ValueHead",
            value_head_kwargs={},
            # Loss configuration.
            baseline_cost=0.25,
            entropy_cost=0.02,
            max_abs_reward=np.inf,
            discount=0.99,
        ),
        train_br=config_dict.create(
            step_key=step_key,
            frame_key=frame_key,
            num_learner_steps=num_steps,
            num_train_arenas=4,
            replay=config_dict.create(
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
            ),
            # Optimizer configuration.
            batch_size=128,
            optimizer_name="adam",
            optimizer_config=config_dict.create(
                # Learning rate: linear schedule.
                learning_rate_init=6e-4,
                learning_rate_end=6e-9,
                learning_rate_steps=100_000,
                # Maximum gradient norm: linear schedule.
                max_norm_init=10.0,
                max_norm_end=5.0,
                max_norm_steps=100_000,
            ),
            # Evaluation.
            eval_frequency=100,
        ),
        transfer=config_dict.create(
            num_planning_steps=num_steps,
            num_learning_steps=num_steps,
            num_train_arenas=4,
            num_planning_arenas=4,
        ),
        world_model=config_dict.create(
            step_key=step_key,
            frame_key=frame_key,
            num_train_arenas=4,
            num_learner_steps=num_steps,
            sequence_length=20,
            sequence_period=None,
            replay_max_size=1_000,
            replay_table_name="world_model",
            # Optimizer.
            batch_size=128,
            learning_rate=3e-4,
            max_gradient_norm=10.0,
            # Loss function.
            reward_cost=10.0,
        ),
    )
    return config


class Scheduler:
    def __init__(self, clients: Mapping[str, Any], config: config_dict.ConfigDict):
        self._clients = clients
        self._config = config

    def run(self):
        with signals.runtime_terminator():
            logging.info("Starting experiment.")

            # 1. Train a BR against a fixed policy.
            logging.info("Training a best-response aganst the fixed policy.")
            for arena in self._clients["br_agent_train_arena"]:
                arena.futures.run()
            self._clients["br_learner"].run(self._config.train_br.num_learner_steps)

            for arena in self._clients["br_agent_train_arena"]:
                arena.stop()

            # 2. Generate a dataset to train a world-model using the frozen BR and fixed policy.
            # 3. Train a world-model using the dataset.
            logging.info("Training a world-model with the frozen BR and the fixed policy.")
            episodes_per_arena = self._config.world_model.replay_max_size / 100
            # TODO(maxsmith): Parallel arenas, but how to get the futures to join when the functions
            # return None?
            self._clients["world_arena"].run(num_episodes=episodes_per_arena)

            self._clients["world_learner"].run(self._config.world_model.num_learner_steps)

            # 4. Warm start a BR(BR) using the frozen world-model.
            logging.info("Warm-starting BR(BR) through planning with a world model.")
            for arena in self._clients["agent_planning_arena"]:
                arena.futures.run()
            self._clients["learner"].run(self._config.transfer.num_planning_steps)

            for arena in self._clients["agent_planning_arena"]:
                arena.stop()

            # 5. Fine tune the BR(BR) using the true world.
            logging.info("Fine-tunning BR(BR) by learning in the true world.")
            for arena in self._clients["agent_train_arena"]:
                arena.futures.run()
            self._clients["learner"].run(self._config.transfer.num_learning_steps)

            for arena in self._clients["agent_train_arena"]:
                arena.stop()

            # 6. Optionally, train a baseline BR(BR) without warm starting.
            logging.info("Training a baseline best-response aganst the best-response policy.")
            for arena in self._clients["baseline_agent_train_arena"]:
                arena.futures.run()
            self._clients["learner"].run(self._config.train_br.num_learner_steps)

            for arena in self._clients["baseline_agent_train_arena"]:
                arena.stop()

            logging.info("Experiment finished.")
            lp.stop()


def run(config: Optional[config_dict.ConfigDict] = None, exist_ok: bool = False, overwrite: bool = True):
    """Build and then run (see `Scheduler`) an one-step transfer experiment."""
    raise NotImplementedError("Currently arenas are set to run automatically.")

    # Load the experiment's config.
    if not config:
        config = get_config()
    # Apply overrides that may optionally be specified via command line.
    if FLAGS.overrides:
        config = flag_utils.apply_overrides(overrides=FLAGS.overrides, base=config)
    if FLAGS.result_dir:
        config.result_dir = FLAGS.result_dir
    config.lock()
    logging.info("Experiment's configuration:\n %s", config)

    result_dir = utils.ResultDirectory(config.result_dir, exist_ok, overwrite=overwrite)
    ujson.dump(config.to_dict(), open(result_dir.file("config.json"), "w"))

    key_sequence = hk.PRNGSequence(config.seed)
    program = lp.Program(name="experiment")
    handles = {}

    # ==============================================================================================
    # 1. Train a BR against a fixed policy.
    game = builders.build_game()
    env_spec = spec_utils.make_game_specs(game)[0]
    agent_graphs = build_services.build_computational_graphs(config.agent, env_spec)

    with program.group("br_agent_replay"):
        handles["br_agent_replay"] = program.add_node(
            builders.build_agent_replay(
                config=config.train_br.replay,
                env_spec=spec_utils.make_game_specs(game)[0],
                state_and_extra_spec=agent_graphs.state_spec,
            )
        )

    with program.group("br_counter"):
        counter_handle = program.add_node(builders.build_counter())

    with program.group("br_learner"):
        handles["br_learner"] = program.add_node(
            builders.build_agent_learner(
                config=config.train_br,
                random_key=next(key_sequence),
                loss_graph=agent_graphs.loss,
                replay=handles["br_agent_replay"],
                counter=counter_handle,
                result_dir=result_dir.make_subdir(program._current_group),
            )
        )

    with program.group("br_agent_train_arena"):
        handles["br_agent_train_arena"] = []
        for i in range(config.train_br.num_train_arenas):
            handles["br_agent_train_arena"].append(
                program.add_node(
                    builders.build_agent_training_arena(
                        config=config.train_br,
                        random_key=next(key_sequence),
                        policy_graph=agent_graphs.policy,
                        initial_state_graph=agent_graphs.initial_state,
                        learner=handles["br_learner"],
                        counter=counter_handle,
                        replay=handles["br_agent_replay"],
                        game=game,
                        bots={1: bots_lib.RandomIntAction(num_actions=8)},
                        result_dir=result_dir.make_subdir(f"{program._current_group}_{i}"),
                    )
                )
            )

    # ==============================================================================================
    # 2. Generate a dataset to train a world-model using the frozen BR and fixed policy.
    # 3. Train a world-model using the dataset.
    world_model = graphs.build_world_graphs(config.world_model, env_spec)

    with program.group("world_reverb"):
        reverb_handle = program.add_node(
            builders.build_world_replay(config.world_model, spec_utils.make_game_specs(game)[0])
        )

    with program.group("world_counter"):
        counter_handle = program.add_node(builders.build_counter())

    with program.group("world_arena"):
        bots = {1: bots_lib.RandomIntAction(num_actions=8)}
        # Player 0 needs to be built on the node to ensure that it can have its variable source built.
        player0_ctor = functools.partial(
            services.EvaluationPolicy,
            policy_fn=agent_graphs.policy,
            initial_state_fn=agent_graphs.initial_state,
            random_key=next(key_sequence),
        )

        handles["world_arena"] = program.add_node(
            builders.build_world_train_arena(
                config.world_model, game, bots, player0_ctor, handles["br_learner"], reverb_handle
            )
        )

    with program.group("world_learner"):
        handles["world_learner"] = program.add_node(
            builders.build_world_model_learner(
                config.world_model,
                next(key_sequence),
                world_model.loss,
                counter_handle,
                reverb_handle,
                result_dir.make_subdir(program._current_group),
            )
        )

    # ==============================================================================================
    # 4. Warm start a BR(BR) using the frozen world-model.
    # 5. Fine tune the BR(BR) using the true world.

    with program.group("agent_replay"):
        handles["agent_replay"] = program.add_node(
            builders.build_agent_replay(
                config=config.train_br.replay,
                env_spec=spec_utils.make_game_specs(game)[0],
                state_and_extra_spec=agent_graphs.state_spec,
            )
        )

    with program.group("counter"):
        counter_handle = program.add_node(builders.build_counter())

    with program.group("learner"):
        handles["learner"] = program.add_node(
            builders.build_agent_learner(
                config=config.train_br,
                random_key=next(key_sequence),
                loss_graph=agent_graphs.loss,
                replay=handles["agent_replay"],
                counter=counter_handle,
                result_dir=result_dir.make_subdir(program._current_group),
            )
        )

    with program.group("agent_planning_arena"):
        handles["agent_planning_arena"] = []
        for i in range(config.transfer.num_planning_arenas):
            handles["agent_planning_arena"].append(
                program.add_node(
                    builders.build_agent_planning_arena(
                        config=config.train_br,
                        random_key=next(key_sequence),
                        policy_graph=agent_graphs.policy,
                        initial_state_graph=agent_graphs.initial_state,
                        learner=handles["learner"],
                        counter=counter_handle,
                        replay=handles["agent_replay"],
                        game=game,
                        transition_graph=world_model.transition,
                        world_init_graph=world_model.initial_state,
                        world_model_source=handles["world_learner"],
                        opponent_variable_source=handles["br_learner"],
                        result_dir=result_dir.make_subdir(f"{program._current_group}_{i}"),
                    )
                )
            )

    with program.group("agent_train_arena"):
        handles["agent_train_arena"] = []
        for i in range(config.transfer.num_train_arenas):
            handles["agent_train_arena"].append(
                program.add_node(
                    builders.build_agent_vs_agent_training_arena(
                        config=config.train_br,
                        random_key=next(key_sequence),
                        policy_graph=agent_graphs.policy,
                        initial_state_graph=agent_graphs.initial_state,
                        learner=handles["learner"],
                        counter=counter_handle,
                        replay=handles["agent_replay"],
                        game=game,
                        opponent_variable_source=handles["br_learner"],
                        result_dir=result_dir.make_subdir(f"{program._current_group}_{i}"),
                    )
                )
            )

    # ==============================================================================================
    # 6. Optionally, train a baseline BR(BR) without warm starting.
    with program.group("baseline_agent_replay"):
        handles["baseline_agent_replay"] = program.add_node(
            builders.build_agent_replay(
                config=config.train_br.replay,
                env_spec=spec_utils.make_game_specs(game)[0],
                state_and_extra_spec=agent_graphs.state_spec,
            )
        )

    with program.group("baseline_counter"):
        counter_handle = program.add_node(builders.build_counter())

    with program.group("baseline_learner"):
        handles["baseline_learner"] = program.add_node(
            builders.build_agent_learner(
                config=config.train_br,
                random_key=next(key_sequence),
                loss_graph=agent_graphs.loss,
                replay=handles["baseline_agent_replay"],
                counter=counter_handle,
                result_dir=result_dir.make_subdir(program._current_group),
            )
        )

    with program.group("baseline_agent_train_arena"):

        handles["baseline_agent_train_arena"] = []
        for i in range(config.train_br.num_train_arenas):
            handles["baseline_agent_train_arena"].append(
                program.add_node(
                    builders.build_agent_vs_agent_training_arena(
                        config=config.train_br,
                        random_key=next(key_sequence),
                        policy_graph=agent_graphs.policy,
                        initial_state_graph=agent_graphs.initial_state,
                        learner=handles["baseline_learner"],
                        counter=counter_handle,
                        replay=handles["baseline_agent_replay"],
                        game=game,
                        opponent_variable_source=handles["br_learner"],
                        result_dir=result_dir.make_subdir(f"{program._current_group}_{i}"),
                    )
                )
            )

    # ==============================================================================================
    with program.group("scheduler"):
        program.add_node(lp.CourierNode(Scheduler, config=config, clients=handles))

    lp.launch(program, launch_type=lp.LaunchType.LOCAL_MULTI_PROCESSING, terminal="current_terminal")


def main(_):
    """Enables running the file directly through absl, and also running with a config input."""
    run()


if __name__ == "__main__":
    app.run(main)
