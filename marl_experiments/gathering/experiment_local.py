import dataclasses
import functools
import importlib
from typing import Any, Callable, List, Mapping, Optional, Union

import haiku as hk
import jax
import launchpad as lp
import numpy as np
import optax
import reverb
from absl import app, logging

from marl import _types, bots, games, individuals, services, utils, worlds
from marl.agents import impala
from marl.services import arenas, evaluation_policy
from marl.services.arenas import training_arena
from marl.services.replay.reverb import adders as reverb_adders
from marl.utils import loggers, node_utils, signals, spec_utils, wrappers
from marl_experiments.gathering import networks
from marl_experiments.gathering.services import render_arena


@dataclasses.dataclass
class Config:
    """Configuration options for IMPALA."""

    result_dir: str = "/scratch/wellman_root/wellman1/mxsmith/tests/impala"

    seed: int = 42
    discount: float = 0.99
    sequence_length: int = 20
    sequence_period: Optional[int] = None
    variable_update_period: int = 1000
    variable_client_key: str = "network"
    step_key: str = "update_steps"
    frame_key: str = "update_frames"

    # Topology.
    num_training_arenas: int = 1

    # Agent configuration.
    timestep_encoder_ctor: str = "marl_experiments.gathering.networks.MLPTimestepEncoder"
    timestep_encoder_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    memory_core_ctor: str = "marl_experiments.gathering.networks.MemoryLessCore"
    memory_core_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    policy_head_ctor: str = "marl_experiments.gathering.networks.PolicyHead"
    policy_head_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    value_head_ctor: str = "marl_experiments.gathering.networks.ValueHead"
    value_head_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    # Optimizer configuration.
    batch_size: int = 32
    learning_rate: Union[float, optax.Schedule] = 6e-4
    max_gradient_norm: float = 40.0

    # Loss configuration.
    baseline_cost: float = 0.25
    entropy_cost: float = 0.01
    max_abs_reward: float = np.inf

    # Replay options.
    replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE
    replay_max_size: int = 1_000_000
    samples_per_insert: int = 1
    min_size_to_sample: int = 1_000
    max_times_sampled: int = 1
    error_buffer: int = 100
    num_prefetch_threads: Optional[int] = None
    samples_per_insert: Optional[float] = float("inf")
    max_queue_size: int = 100

    # Evaluation.
    render_frequency: int = 100

    def __post_init__(self):
        assert (
            self.max_queue_size > self.batch_size + 1
        ), """
        max_queue_size must be strictly larger than the batch size:
        - during the last step in an episode we might write 2 sequences to
          Reverb at once (that's how SequenceAdder works)
        - Reverb does insertion/sampling in multiple threads, so data is
          added asynchronously at unpredictable times. Therefore we need
          additional buffer size in order to avoid deadlocks."""


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
def build_rendering_arena_node(
    config: Config,
    policy_fn: hk.Transformed,
    initial_state_fn: hk.Transformed,
    random_key: jax.random.PRNGKey,
    learner_update_handle: lp.CourierHandle,
    opponents,
    counter_handle: lp.CourierHandle,
    result_dir: utils.ResultDirectory,
):
    variable_source = services.VariableClient(
        source=learner_update_handle,
        key=config.variable_client_key,
    )
    evaluation_policy = services.EvaluationPolicy(
        policy_fn=policy_fn,
        initial_state_fn=initial_state_fn,
        variable_source=variable_source,
        random_key=random_key,
    )
    local_counter = services.Counter(parent=counter_handle)

    return render_arena.EvaluationArena(
        agents={0: evaluation_policy},
        bots=opponents,
        scenarios=render_arena.EvaluationScenario(game_ctor=build_game, num_episodes=5),
        evaluation_frequency=config.render_frequency,
        counter=local_counter,
        step_key=config.step_key,
        result_dir=result_dir,
    )


def run(config: Optional[Config] = None, exist_ok: bool = False, overwrite: bool = True):
    if not config:
        config = Config()
    result_dir = utils.ResultDirectory(config.result_dir, exist_ok, overwrite=overwrite)

    random_key = jax.random.PRNGKey(config.seed)
    game = build_game()
    env_spec = spec_utils.make_game_specs(game)[0]

    opponents = {1: bots.ConstantIntAction(action=games.GatheringActions.NOOP.value, env_spec=env_spec)}

    program = lp.Program(name="experiment")

    # TODO(maxsmith): Configurable networks and ensure that this is captured in the snapshot.
    graphs = impala.build_computational_graphs(config, env_spec)
    train_policy_graph, loss_graph, initial_state_graph, state_spec, eval_policy_graph = graphs
    snapshot_template = services.Snapshot(
        ctor=impala.build_computational_graphs, ctor_kwargs=dict(config=config, env_spec=env_spec)
    )

    with program.group("counter"):
        counter_handle = program.add_node(impala.build_counter_node())

    with program.group("reverb"):
        reverb_handle = program.add_node(impala.build_reverb_node(config, env_spec, state_spec))

    with program.group("learner_update"):
        random_key, subkey = jax.random.split(random_key)
        learner_update_handle = program.add_node(
            impala.build_update_node(
                config,
                subkey,
                loss_graph,
                reverb_handle,
                counter_handle,
                result_dir.make_subdir(program._current_group),
            )
        )

    with program.group("training_arena"):
        for node_i in range(config.num_training_arenas):
            node_name = f"{program._current_group}_{node_i}"
            random_key, subkey = jax.random.split(random_key)

            program.add_node(
                impala.build_training_arena_node(
                    config,
                    subkey,
                    train_policy_graph,
                    initial_state_graph,
                    reverb_handle,
                    learner_update_handle,
                    counter_handle,
                    game,
                    opponents,
                    result_dir.make_subdir(node_name),
                )
            )

    with program.group("evaluation_arena"):
        # Policy evaluation.
        random_key, subkey = jax.random.split(random_key)
        evaluation_node = impala.build_evaluation_arena_node(
            config,
            train_policy_graph,
            initial_state_graph,
            subkey,
            learner_update_handle,
            opponents,
            counter_handle,
            build_game,
            result_dir.make_subdir(program._current_group),
        )

        # Training rendering.
        train_name = f"{program._current_group}_render_train"
        random_key, subkey = jax.random.split(random_key)
        train_render_node = build_rendering_arena_node(
            config,
            train_policy_graph,
            initial_state_graph,
            subkey,
            learner_update_handle,
            opponents,
            counter_handle,
            result_dir.make_subdir(train_name),
        )

        # Evaluation rendering.
        eval_name = f"{program._current_group}_render_eval"
        random_key, subkey = jax.random.split(random_key)
        eval_render_node = build_rendering_arena_node(
            config,
            eval_policy_graph,
            initial_state_graph,
            subkey,
            learner_update_handle,
            opponents,
            counter_handle,
            result_dir.make_subdir(eval_name),
        )

        program.add_node(lp.MultiThreadingColocation([evaluation_node, train_render_node, eval_render_node]))

    with program.group("saver"):
        program.add_node(
            impala.build_snapshot_node(
                snapshot_template, learner_update_handle, result_dir.make_subdir(program._current_group)
            )
        )

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
