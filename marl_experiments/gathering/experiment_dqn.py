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

from marl import _types
from marl import bots as bots_lib
from marl import games, individuals, layouts, services, utils, worlds
from marl.agents.dqn import dqn
from marl.services import arenas, evaluation_policy
from marl.services.arenas import training_arena
from marl.services.replay.reverb import adders as reverb_adders
from marl.utils import loggers, node_utils, signals, spec_utils, wrappers
from marl_experiments.gathering import build_services, networks
from marl_experiments.gathering.services import render_arena


@dataclasses.dataclass
class DQNConfig:
    """Configuration options for DQN."""

    result_dir: str = "/scratch/wellman_root/wellman1/mxsmith/tests/dqn_long"

    seed: int = 0
    discount: float = 0.99
    sequence_length: int = 20
    sequence_period: Optional[int] = None
    step_key: str = "update_steps"
    frame_key: str = "update_frames"

    # Topology.
    num_training_arenas: int = 4

    # Agent configuration.
    timestep_encoder_ctor: str = "marl_experiments.gathering.networks.MLPTimestepEncoder"
    timestep_encoder_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    memory_core_ctor: str = "marl_experiments.gathering.networks.MemoryLessCore"
    memory_core_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    value_head_ctor: str = "marl_experiments.gathering.networks.ValueHead"
    value_head_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    schedule_name: str = "Linear"
    schedule_kwargs: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: dict(x_initial=1.0, x_final=0.3, num_steps=1_000_000)
    )

    # Optimizer configuration.
    batch_size: int = 32
    learning_rate: Union[float, optax.Schedule] = 6e-4
    adam_momentum_decay: float = 0.0
    adam_variance_decay: float = 0.99
    adam_eps: float = 1e-8
    adam_eps_root: float = 0.0
    max_gradient_norm: float = 40.0

    # Loss configuration.
    max_abs_reward: float = np.inf

    # Replay options.
    replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE
    replay_max_size: int = 1_000_000
    samples_per_insert: int = 1.0
    min_size_to_sample: int = 10_000
    max_times_sampled: int = 1
    error_buffer: int = 100
    num_prefetch_threads: Optional[int] = None
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


def str_to_class(path: str) -> Any:
    tokens = path.split(".")
    module = ".".join(tokens[:-1])
    name = tokens[-1]
    module = importlib.import_module(module)
    return getattr(module, name)


def build_computational_graphs(config: DQNConfig, env_spec: worlds.EnvironmentSpec):
    timestep = spec_utils.zeros_from_spec(env_spec)

    def _dqn_graphs():
        num_actions = env_spec.action.num_values
        timestep_encoder = str_to_class(config.timestep_encoder_ctor)(
            num_actions=num_actions, **config.timestep_encoder_kwargs
        )
        memory_core = str_to_class(config.memory_core_ctor)(**config.memory_core_kwargs)
        value_head = str_to_class(config.value_head_ctor)(num_actions=num_actions, **config.value_head_kwargs)
        epsilon_schedule = str_to_class(f"marl.utils.schedules.{config.schedule_name}")(**config.schedule_kwargs)

        dqn_kwargs = dict(
            # Sub-modules.
            timestep_encoder=timestep_encoder,
            memory_core=memory_core,
            value_head=value_head,
            # Hyperparameters.
            discount=config.discount,
            max_abs_reward=config.max_abs_reward,
            epsilon=epsilon_schedule,
        )

        train_dqn = dqn.DQN(evaluation=False, **dqn_kwargs)
        eval_dqn = dqn.DQN(evaluation=True, **dqn_kwargs)

        def init():
            return train_dqn(timestep, train_dqn.initial_state(None))

        return init, (
            train_dqn.__call__,
            train_dqn.loss,
            train_dqn.initial_state,
            train_dqn.state_spec,
            eval_dqn.__call__,
        )

    hk_graphs = hk.multi_transform(_dqn_graphs)
    train_policy = hk.Transformed(init=hk_graphs.init, apply=hk_graphs.apply[0])
    loss = hk.Transformed(init=hk_graphs.init, apply=hk_graphs.apply[1])
    initial_state = hk.Transformed(init=hk_graphs.init, apply=hk_graphs.apply[2])
    state_spec = hk.without_apply_rng(hk_graphs).apply[3](None)  # No parameters.
    eval_policy = hk.Transformed(init=hk_graphs.init, apply=hk_graphs.apply[4])
    return train_policy, loss, initial_state, state_spec, eval_policy


@node_utils.build_courier_node
def build_snapshot_node(
    snapshot_template: services.Snapshot,
    learner_update_handle: lp.CourierHandle,
    result_dir: utils.ResultDirectory,
):
    return services.Snapshotter(
        variable_source=learner_update_handle,
        snapshot_templates={"dqn": snapshot_template},
        directory=result_dir.dir,
        max_to_keep=2,
    )


@node_utils.build_courier_node
def build_counter_node():
    return services.Counter()


def run(config: Optional[DQNConfig] = None, exist_ok: bool = False, overwrite: bool = True):
    if not config:
        config = DQNConfig()
    result_dir = utils.ResultDirectory(config.result_dir, exist_ok, overwrite=overwrite)

    random_key = jax.random.PRNGKey(config.seed)
    game = build_game()
    env_spec = spec_utils.make_game_specs(game)[0]

    bots = {1: bots_lib.ConstantIntAction(action=games.GatheringActions.NOOP.value, env_spec=env_spec)}

    program = lp.Program(name="experiment")

    # TODO(maxsmith): Configurable networks and ensure that this is captured in the snapshot.
    graphs = build_computational_graphs(config, env_spec)
    train_policy, loss, initial_state, state_spec, eval_policy = graphs
    snapshot_template = services.Snapshot(
        ctor=build_computational_graphs, ctor_kwargs=dict(config=config, env_spec=env_spec)
    )

    # Build partial constructors for all of the program nodes, excluding the handles to other nodes.
    # The handles will be provided by kwargs during program layout.
    replay_ctor = functools.partial(
        build_services.build_reverb_node,
        config=config,
        env_spec=env_spec,
        state_and_extra_spec=state_spec,
    )
    learner_ctor = functools.partial(build_services.build_learner_node, config=config, loss_graph=loss)
    train_arena_ctor = functools.partial(
        build_services.build_training_arena_node,
        config=config,
        policy_graph=train_policy,
        initial_state_graph=initial_state,
        game=game,
        bots=bots,
    )
    eval_arena_ctors = []
    eval_arena_ctors.append(
        functools.partial(
            build_services.build_evaluation_arena_node,
            config=config,
            policy_graph=eval_policy,
            initial_state_graph=initial_state,
            bots=bots,
            game_ctor=build_game,
        )
    )

    program = layouts.build_distributed_training_program(
        program=program,
        counter_ctor=build_counter_node,
        replay_ctor=replay_ctor,
        learner_ctor=learner_ctor,
        train_arena_ctor=train_arena_ctor,
        eval_arena_ctors=eval_arena_ctors,
        result_dir=result_dir,
        seed=config.seed,
    )

    lp.launch(
        program,
        launch_type=lp.LaunchType.LOCAL_MULTI_PROCESSING,
        terminal="current_terminal",
    )


def main(_):
    """Enables running the file directly through absl, and also running with a config input."""
    run()


if __name__ == "__main__":
    app.run(main)
