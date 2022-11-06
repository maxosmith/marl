"""Utility functions for building common services or modules."""
import functools
from typing import List, Optional

import haiku as hk
import jax
import launchpad as lp
import optax
import reverb

from marl import services, utils, worlds
from marl.agents import impala
from marl.services.replay.reverb import adders as reverb_adders
from marl.utils import import_utils, loggers, node_utils, spec_utils
from marl_experiments.roshambo.services import evaluation_arena, training_arena


def build_impala_graphs(config, env_spec: worlds.EnvironmentSpec):
    timestep = spec_utils.zeros_from_spec(env_spec)

    def _impala_graphs():
        num_actions = 3
        config.policy_head_kwargs["num_actions"] = num_actions

        timestep_encoder = import_utils.initialize(config.timestep_encoder_ctor, config.timestep_encoder_kwargs)
        memory_core = import_utils.initialize(config.memory_core_ctor, config.memory_core_kwargs)
        policy_head = import_utils.initialize(config.policy_head_ctor, config.policy_head_kwargs)
        value_head = import_utils.initialize(config.value_head_ctor, config.value_head_kwargs)

        impala_kwargs = dict(
            # Sub-modules.
            timestep_encoder=timestep_encoder,
            memory_core=memory_core,
            policy_head=policy_head,
            value_head=value_head,
            # Hyperparameters.
            discount=config.discount,
            max_abs_reward=config.max_abs_reward,
            baseline_cost=config.baseline_cost,
            entropy_cost=config.entropy_cost,
        )

        train_impala = impala.IMPALA(evaluation=False, **impala_kwargs)
        eval_impala = impala.IMPALA(evaluation=True, **impala_kwargs)

        def init():
            return train_impala(timestep, train_impala.initial_state(None))

        return init, (
            train_impala.__call__,
            train_impala.loss,
            train_impala.initial_state,
            train_impala.state_spec,
            eval_impala.__call__,
        )

    hk_graphs = hk.multi_transform(_impala_graphs)
    train_policy = hk.Transformed(init=hk_graphs.init, apply=hk_graphs.apply[0])
    loss = hk.Transformed(init=hk_graphs.init, apply=hk_graphs.apply[1])
    initial_state = hk.Transformed(init=hk_graphs.init, apply=hk_graphs.apply[2])
    state_spec = hk.without_apply_rng(hk_graphs).apply[3](None)  # No parameters.
    eval_policy = hk.Transformed(init=hk_graphs.init, apply=hk_graphs.apply[4])
    return impala.IMPALAGraphs(
        policy=train_policy, loss=loss, initial_state=initial_state, state_spec=state_spec, eval_policy=eval_policy
    )


@node_utils.build_courier_node
def build_counter_node():
    return services.Counter()


@node_utils.build_courier_node
def build_learner_node(
    loss_graph: hk.Transformed,
    replay: lp.CourierHandle,
    counter: lp.CourierHandle,
    step_key: str,
    frame_key: str,
    random_key: jax.random.PRNGKey,
    result_dir: utils.ResultDirectory,
    *,
    replay_table_name: str,
    learning_rate: float,
    max_gradient_norm: float,
    batch_size: int,
):
    logger = loggers.LoggerManager(
        loggers=[
            loggers.TerminalLogger(time_frequency=5),
            loggers.TensorboardLogger(result_dir.dir, step_key=step_key),
        ],
        time_frequency=5,  # Seconds.
    )
    local_counter = services.Counter(parent=counter)

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_gradient_norm),
        optax.adam(learning_rate),
    )

    data_iterator = services.ReverbPrefetchClient(
        reverb_client=replay,
        table_name=replay_table_name,
        batch_size=batch_size,
    )
    return services.LearnerUpdate(
        loss_fn=loss_graph,
        optimizer=optimizer,
        data_iterator=data_iterator,
        logger=logger,
        counter=local_counter,
        step_key=step_key,
        frame_key=frame_key,
        random_key=random_key,
    )


@node_utils.build_courier_node(disable_run=True)
def build_training_arena_node(
    policy_graph: hk.Transformed,
    initial_state_graph: hk.Transformed,
    replay: lp.CourierHandle,
    learner: lp.CourierHandle,
    counter: lp.CourierHandle,
    random_key: jax.random.PRNGKey,
    game: worlds.Game,
    bots,
    replay_table_name: str,
    sequence_length: int,
    variable_update_period: int,
    step_key: str,
    result_dir: utils.ResultDirectory,
    sequence_period: Optional[int] = None,
):
    """Build a training arena: containing a learner and logging its' experiences.

    Args:
        variable_client_key: Variable name to request from the learner (variable source).
        variable_update_period: Interval between fetches, specified as the number of calls to the
            variable update() method.
        step_key:
    """
    # NOTE: That the last transition in the sequence is used for bootstrapping
    # only and is ignored otherwise. So we need to make sure that sequences
    # overlap on one transition, thus "-1" in the period length computation.
    reverb_adder = reverb_adders.SequenceAdder(
        client=replay,
        priority_fns={replay_table_name: None},
        period=sequence_period or (sequence_length - 1),
        sequence_length=sequence_length,
    )
    # Variable client is responsible for syncing with the Updating node, but does not tell
    # that node when it should update.
    variable_source = services.VariableClient(
        source=learner,
        update_period=variable_update_period,
    )
    policy = services.LearnerPolicy(
        policy_fn=policy_graph,
        initial_state_fn=initial_state_graph,
        reverb_adder=reverb_adder,
        variable_source=variable_source,
        random_key=random_key,
        backend="cpu",
    )
    local_counter = services.Counter(parent=counter)

    # TODO(maxsmith): Currently this assumes the learner is always player 0, generalize.
    players = bots
    players[0] = policy

    logger = loggers.TensorboardLogger(result_dir.dir, step_key=step_key)
    train_arena = training_arena.TrainingArena(
        game=game,
        players=players,
        logger=logger,
        counter=local_counter,
        step_key=step_key,
    )
    return train_arena


@node_utils.build_courier_node
def build_evaluation_arena_node(
    policy_graph: hk.Transformed,
    initial_state_graph: hk.Transformed,
    random_key: jax.random.PRNGKey,
    learner: lp.CourierHandle,
    bots,
    counter: lp.CourierHandle,
    game_ctor,
    evaluation_frequency: int,
    variable_client_key: str,
    step_key: str,
    result_dir: utils.ResultDirectory,
):
    variable_source = services.VariableClient(
        source=learner,
        key=variable_client_key,
    )
    evaluation_policy = services.EvaluationPolicy(
        policy_fn=policy_graph,
        initial_state_fn=initial_state_graph,
        variable_source=variable_source,
        random_key=random_key,
    )
    logger = loggers.TensorboardLogger(result_dir.dir, step_key=step_key)
    local_counter = services.Counter(parent=counter)

    return evaluation_arena.EvaluationArena(
        agents={0: evaluation_policy},
        bots=bots,
        scenarios=evaluation_arena.EvaluationScenario(game_ctor=game_ctor, num_episodes=5),
        evaluation_frequency=evaluation_frequency,
        counter=local_counter,
        logger=logger,
        step_key=step_key,
    )


def build_reverb_node(
    env_spec: worlds.EnvironmentSpec,
    state_and_extra_spec: worlds.TreeSpec,
    *,
    samples_per_insert: float,
    min_size_to_sample: int,
    max_times_sampled: int,
    error_buffer: int,
    replay_max_size: int,
    sequence_length: int,
    replay_table_name: str,
):
    """."""

    def _build_reverb_node(
        env_spec: worlds.EnvironmentSpec, sequence_length: int, table_name: str
    ) -> List[reverb.Table]:
        signature = reverb_adders.SequenceAdder.signature(
            env_spec,
            state_and_extra_spec,
            sequence_length=sequence_length,
        )
        rate_limiter = reverb.rate_limiters.SampleToInsertRatio(
            samples_per_insert=samples_per_insert,
            min_size_to_sample=min_size_to_sample,
            error_buffer=error_buffer,
        )
        replay_table = reverb.Table(
            name=table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=replay_max_size,
            max_times_sampled=max_times_sampled,
            rate_limiter=rate_limiter,
            signature=signature,
        )
        return [replay_table]

    build_reverb_node_fn = functools.partial(
        _build_reverb_node,
        env_spec=env_spec,
        sequence_length=sequence_length,
        table_name=replay_table_name,
    )
    return lp.ReverbNode(build_reverb_node_fn)
