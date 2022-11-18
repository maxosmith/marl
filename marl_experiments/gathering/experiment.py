import dataclasses
import functools
from typing import Any, Mapping, Optional

import haiku as hk
import jax
import launchpad as lp
import numpy as np
import ujson
from absl import app, flags, logging
from ml_collections import config_dict

from marl import bots as bots_lib
from marl import games, layouts, services, utils
from marl.agents import impala
from marl.services.replay.reverb import adders as reverb_adders
from marl.utils import flag_utils, node_utils, spec_utils, wrappers
from marl_experiments.gathering import build_services
from marl_experiments.gathering.services import render_arena

flag_utils.common_experiment_flags()
flags.adopt_module_key_flags(flag_utils)
FLAGS = flags.FLAGS


def get_config() -> config_dict.ConfigDict:
    """Default configuration for this experiment."""
    config = config_dict.create(
        result_dir="/scratch/wellman_root/wellman1/mxsmith/results/marl/gathering/impala/",
        seed=42,
        discount=0.99,
        sequence_length=20,
        sequence_period=None,
        step_key="learner_steps",
        frame_key="learner_frames",
        # Termination condition.
        max_steps=4_000,  # Roughly one hour of walltime.
        # Topology.
        num_training_arenas=4,
        # Agent configuration.
        timestep_encoder_ctor="marl_experiments.gathering.networks.MLPTimestepEncoder",
        timestep_encoder_kwargs={},
        memory_core_ctor="marl_experiments.gathering.networks.MemoryLessCore",
        memory_core_kwargs={},
        policy_head_ctor="marl_experiments.gathering.networks.PolicyHead",
        policy_head_kwargs={},
        value_head_ctor="marl_experiments.gathering.networks.ValueHead",
        value_head_kwargs={},
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
        # Loss configuration.
        baseline_cost=0.25,
        entropy_cost=0.02,
        max_abs_reward=np.inf,
        # Replay options.
        replay_table_name=reverb_adders.DEFAULT_PRIORITY_TABLE,
        replay_max_size=1_000_000,
        samples_per_insert=1.0,
        min_size_to_sample=1,
        max_times_sampled=1,
        error_buffer=100,
        num_prefetch_threads=None,
        max_queue_size=500,
        # Evaluation.
        render_frequency=100,
    )
    return config


def validate_config(config: config_dict.ConfigDict):
    """Validates that the config has valid settings."""
    assert (
        config.max_queue_size > config.batch_size + 1
    ), """
        max_queue_size must be strictly larger than the batch size:
        - during the last step in an episode we might write 2 sequences to
            Reverb at once (that's how SequenceAdder works)
        - Reverb does insertion/sampling in multiple threads, so data is
            added asynchronously at unpredictable times. Therefore we need
            additional buffer size in order to avoid deadlocks."""
    assert config.optimizer_name in ["adam", "rmsprop"]


def build_game():
    game = games.Gathering(
        n_agents=2,
        map_name="default_small",
        global_observation=False,
        viewbox_width=10,
        viewbox_depth=10,
    )
    game = wrappers.TimeLimit(game, num_steps=100)
    return game


@node_utils.build_courier_node
def build_steps_limiter(
    counter: lp.CourierHandle,
    max_steps: int,
    step_key: str,
) -> services.StepsLimiter:
    """Limit the program execution based on the number of"""
    local_counter = services.Counter(parent=counter)
    return services.StepsLimiter(counter=local_counter, max_steps=max_steps, step_key=step_key)


@node_utils.build_courier_node
def build_rendering_arena_node(
    config: config_dict.ConfigDict,
    policy_graph: hk.Transformed,
    initial_state_graph: hk.Transformed,
    random_key: jax.random.PRNGKey,
    learner: lp.CourierHandle,
    bots,
    counter: lp.CourierHandle,
    result_dir: utils.ResultDirectory,
):
    variable_source = services.VariableClient(
        source=learner,
        key=config.variable_client_key,
    )
    evaluation_policy = services.EvaluationPolicy(
        policy_fn=policy_graph,
        initial_state_fn=initial_state_graph,
        variable_source=variable_source,
        random_key=random_key,
    )
    local_counter = services.Counter(parent=counter)

    return render_arena.EvaluationArena(
        agents={0: evaluation_policy},
        bots=bots,
        scenarios=render_arena.EvaluationScenario(game_ctor=build_game, num_episodes=5),
        evaluation_frequency=config.render_frequency,
        counter=local_counter,
        step_key=config.step_key,
        result_dir=result_dir,
    )


def run(config: Optional[config_dict.ConfigDict] = None, exist_ok: bool = False, overwrite: bool = True):
    # Load the experiment's config.
    if not config:
        config = get_config()
    # Apply overrides that may optionally be specified via command line.
    if FLAGS.overrides:
        config = flag_utils.apply_overrides(overrides=FLAGS.overrides, base=config)
    if FLAGS.result_dir:
        config.result_dir = FLAGS.result_dir
    config.lock()
    validate_config(config)
    logging.info("Experiment's configuration:\n %s", config)

    # Set-up the directory for saving all results and log the experiment's configuration.
    result_dir = utils.ResultDirectory(config.result_dir, exist_ok, overwrite=overwrite)
    ujson.dump(config.to_dict(), open(result_dir.file("config.json"), "w"))

    game = build_game()
    env_spec = spec_utils.make_game_specs(game)[0]

    bots = {1: bots_lib.RandomIntAction(num_actions=env_spec.action.num_values)}

    program = lp.Program(name="experiment")

    # TODO(maxsmith): Configurable networks and ensure that this is captured in the snapshot.
    graphs = build_services.build_computational_graphs(config, env_spec)
    snapshot_template = services.Snapshot(
        ctor=build_services.build_computational_graphs, ctor_kwargs=dict(config=config, env_spec=env_spec)
    )

    # Build partial constructors for all of the program nodes, excluding the handles to other nodes.
    # The handles will be provided by kwargs during program layout.
    replay_ctor = functools.partial(
        build_services.build_reverb_node,
        config=config,
        env_spec=env_spec,
        state_and_extra_spec=graphs.state_spec,
    )
    learner_ctor = functools.partial(
        build_services.build_learner_node,
        config=config,
        loss_graph=graphs.loss,
        optimizer_name=config.optimizer_name,
        optimizer_config=config.optimizer_config,
    )
    train_arena_ctor = functools.partial(
        build_services.build_training_arena_node,
        config=config,
        policy_graph=graphs.policy,
        initial_state_graph=graphs.initial_state,
        game=game,
        bots=bots,
    )
    eval_arena_ctors = []
    eval_arena_ctors.append(
        functools.partial(
            build_services.build_evaluation_arena_node,
            config=config,
            policy_graph=graphs.eval_policy,
            initial_state_graph=graphs.initial_state,
            bots=bots,
            game_ctor=build_game,
        )
    )

    # Layout the main nodes as a distributed offline RL learning program.
    program = layouts.build_distributed_training_program(
        program=program,
        counter_ctor=impala.build_counter_node,
        replay_ctor=replay_ctor,
        learner_ctor=learner_ctor,
        train_arena_ctor=train_arena_ctor,
        eval_arena_ctors=eval_arena_ctors,
        num_train_arenas=config.num_training_arenas,
        result_dir=result_dir,
        seed=config.seed,
    )

    # Add additional nodes to the program.
    with program.group("steps_limiter"):
        program.add_node(
            build_steps_limiter(
                counter=program.groups["counter"][0].create_handle(),
                max_steps=config.max_steps,
                step_key=config.step_key,
            )
        )

    # with program.group("saver"):
    #     program.add_node(
    #         impala.build_snapshot_node(
    #             snapshot_template, learner_update_handle, result_dir.make_subdir(program._current_group)
    #         )
    #     )

    lp.launch(
        program,
        launch_type=lp.LaunchType.LOCAL_MULTI_PROCESSING,
        # launch_type=lp.LaunchType.LOCAL_MULTI_THREADING,
        terminal="current_terminal",
    )


def main(_):
    """Enables running the file directly through absl, and also running with a config input."""
    run()


if __name__ == "__main__":
    app.run(main)
