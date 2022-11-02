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
from marl.agents.dqn import dqn
from marl.services import arenas, evaluation_policy
from marl.services.arenas import training_arena
from marl.services.replay.reverb import adders as reverb_adders
from marl.utils import loggers, node_utils, signals, spec_utils, wrappers
from marl_experiments.gathering import networks
from marl_experiments.gathering.services import render_arena


@dataclasses.dataclass
class DQNConfig:
    """Configuration options for DQN."""

    result_dir: str = "/scratch/wellman_root/wellman1/mxsmith/tests/dqn_long"

    seed: int = 0
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
    samples_per_insert: int = 4
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
    game = games.Gathering(n_agents=2, map_name="default_small", global_observation=True)
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


def build_reverb_node(config: DQNConfig, env_spec: worlds.EnvironmentSpec, state_spec: worlds.TreeSpec):
    def _build_reverb_node(
        env_spec: worlds.EnvironmentSpec, sequence_length: int, table_name: str
    ) -> List[reverb.Table]:
        signature = reverb_adders.SequenceAdder.signature(
            env_spec,
            state_spec,
            sequence_length=sequence_length,
        )
        rate_limiter = reverb.rate_limiters.SampleToInsertRatio(
            samples_per_insert=config.samples_per_insert,
            min_size_to_sample=config.min_size_to_sample,
            error_buffer=config.error_buffer,
        )
        replay_table = reverb.Table(
            name=table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=config.replay_max_size,
            max_times_sampled=config.max_times_sampled,
            rate_limiter=rate_limiter,
            signature=signature,
        )
        return [replay_table]

    build_reverb_node_fn = functools.partial(
        _build_reverb_node,
        env_spec=env_spec,
        sequence_length=config.sequence_length,
        table_name=config.replay_table_name,
    )
    return lp.ReverbNode(build_reverb_node_fn)


@node_utils.build_courier_node
def build_update_node(
    config: DQNConfig,
    random_key: jax.random.PRNGKey,
    loss_graph: hk.Transformed,
    reverb_handle: lp.CourierHandle,
    counter_handle: lp.CourierHandle,
    result_dir: utils.ResultDirectory,
):
    logger = loggers.LoggerManager(
        loggers=[
            loggers.TerminalLogger(time_frequency=5),
            loggers.TensorboardLogger(result_dir.dir, step_key=config.step_key),
        ],
        time_frequency=5,  # Seconds.
    )
    local_counter = services.Counter(parent=counter_handle)

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_gradient_norm),
        optax.adam(
            config.learning_rate,
            b1=config.adam_momentum_decay,
            b2=config.adam_variance_decay,
            eps=config.adam_eps,
            eps_root=config.adam_eps_root,
        ),
    )

    data_iterator = services.ReverbPrefetchClient(
        reverb_client=reverb_handle,
        table_name=config.replay_table_name,
        batch_size=config.batch_size,
    )
    return services.LearnerUpdate(
        loss_fn=loss_graph,
        optimizer=optimizer,
        data_iterator=data_iterator,
        logger=logger,
        counter=local_counter,
        step_key=config.step_key,
        frame_key=config.frame_key,
        random_key=random_key,
    )


@node_utils.build_courier_node(disable_run=False)
def build_training_arena_node(
    config: DQNConfig,
    random_key: jax.random.PRNGKey,
    policy_graph: hk.Transformed,
    initial_state_graph: hk.Transformed,
    reverb_handle: lp.CourierHandle,
    learner_update_handle: lp.CourierHandle,
    counter_handle: lp.CourierHandle,
    game: worlds.Game,
    players,
    result_dir: utils.ResultDirectory,
):
    # NOTE: Currently LaunchPad does not support RPC methods receiving np.Arrays.
    # Therefore, we cannot send TimeSteps over RPC, so instead each learner must
    # be directly associated with a training arena.
    #
    # NOTE: That the last transition in the sequence is used for bootstrapping
    # only and is ignored otherwise. So we need to make sure that sequences
    # overlap on one transition, thus "-1" in the period length computation.
    reverb_adder = reverb_adders.SequenceAdder(
        client=reverb_handle,
        priority_fns={config.replay_table_name: None},
        period=config.sequence_period or (config.sequence_length - 1),
        sequence_length=config.sequence_length,
    )
    # Variable client is responsible for syncing with the Updating node, but does not tell
    # that node when it should update.
    variable_source = services.VariableClient(
        source=learner_update_handle,
        key=config.variable_client_key,
        update_period=config.variable_update_period,
    )
    policy = services.LearnerPolicy(
        policy_fn=policy_graph,
        initial_state_fn=initial_state_graph,
        reverb_adder=reverb_adder,
        variable_source=variable_source,
        random_key=random_key,
        backend="cpu",
    )
    local_counter = services.Counter(parent=counter_handle)

    players[0] = policy

    logger = loggers.TensorboardLogger(result_dir.dir, step_key=config.step_key)
    train_arena = arenas.TrainingArena(
        game=game,
        players=players,
        logger=logger,
        counter=local_counter,
        step_key=config.step_key,
    )
    return train_arena


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


@node_utils.build_courier_node
def build_evaluation_arena_node(
    config: DQNConfig,
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
    logger = loggers.TensorboardLogger(result_dir.dir, step_key=config.step_key)
    local_counter = services.Counter(parent=counter_handle)

    return arenas.EvaluationArena(
        agents={0: evaluation_policy},
        bots=opponents,
        scenarios=arenas.evaluation_arena.EvaluationScenario(game_ctor=build_game, num_episodes=5),
        evaluation_frequency=config.render_frequency,
        counter=local_counter,
        logger=logger,
        step_key=config.step_key,
    )


@node_utils.build_courier_node
def build_rendering_arena_node(
    config: DQNConfig,
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


def run(config: Optional[DQNConfig] = None, exist_ok: bool = False, overwrite: bool = True):
    if not config:
        config = DQNConfig()
    result_dir = utils.ResultDirectory(config.result_dir, exist_ok, overwrite=overwrite)

    random_key = jax.random.PRNGKey(config.seed)
    game = build_game()
    env_spec = spec_utils.make_game_specs(game)[0]

    opponents = {1: bots.ConstantIntAction(action=games.GatheringActions.NOOP.value, env_spec=env_spec)}

    program = lp.Program(name="experiment")

    # TODO(maxsmith): Configurable networks and ensure that this is captured in the snapshot.
    graphs = build_computational_graphs(config, env_spec)
    train_policy_graph, loss_graph, initial_state_graph, state_spec, eval_policy_graph = graphs
    snapshot_template = services.Snapshot(
        ctor=build_computational_graphs, ctor_kwargs=dict(config=config, env_spec=env_spec)
    )

    with program.group("counter"):
        counter_handle = program.add_node(build_counter_node())

    with program.group("reverb"):
        reverb_handle = program.add_node(build_reverb_node(config, env_spec, state_spec))

    with program.group("learner_update"):
        random_key, subkey = jax.random.split(random_key)
        learner_update_handle = program.add_node(
            build_update_node(
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
                build_training_arena_node(
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
        evaluation_node = build_evaluation_arena_node(
            config,
            train_policy_graph,
            initial_state_graph,
            subkey,
            learner_update_handle,
            opponents,
            counter_handle,
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
            build_snapshot_node(
                snapshot_template, learner_update_handle, result_dir.make_subdir(program._current_group)
            )
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
