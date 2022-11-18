import dataclasses
import functools
from typing import List, Optional, Union

import haiku as hk
import jax
import launchpad as lp
import numpy as np
import optax
import reverb
from absl import app, logging

from marl import bots, games, services, utils, worlds
from marl.services.replay.reverb import adders as reverb_adders
from marl.utils import loggers, node_utils, spec_utils, wrappers
from marl.utils.loggers import terminal as terminal_logger_lib
from marl_experiments.gathering import networks
from marl_experiments.gathering.services import model_dataset_arena, supervised_learning
from marl_experiments.gathering.world_model import WorldModel


@dataclasses.dataclass
class ModelTrainConfig:

    result_dir: str = "/scratch/wellman_root/wellman1/mxsmith/results/marl/gathering/test_model_train"
    step_key: str = "step"
    frame_key: str = "frame"

    seed: int = 42

    replay_table_name = "world"
    replay_max_size = 1_000_000
    sequence_length: int = 20
    sequence_period: Optional[int] = None

    reward_cost = 0.0
    discount: float = 0.99

    # Optimizer configuration.
    batch_size: int = 32
    learning_rate: Union[float, optax.Schedule] = 6e-4
    adam_momentum_decay: float = 0.0
    adam_variance_decay: float = 0.99
    adam_eps: float = 1e-8
    adam_eps_root: float = 0.0
    max_gradient_norm: float = 40.0


def build_computational_graphs(config: ModelTrainConfig, env_spec: worlds.EnvironmentSpec):
    dummy_input = dict(
        world_state=spec_utils.zeros_from_spec(env_spec.observation),
        actions=np.zeros([2], dtype=np.int32),
    )

    def _world_model_graphs():
        world_model = WorldModel(
            state_shape=env_spec.observation.shape,
            input_encoder=networks.WorldStateLinearEncoder(
                state_shape=env_spec.observation.shape, num_actions=env_spec.action.num_values
            ),
            memory_core=networks.MemoryCore(),
            state_prediction_head=networks.WorldStateLinearPredictionHead(state_shape=env_spec.observation.shape),
            reward_prediction_head=networks.RewardPredictionHead(),
            reward_cost=config.reward_cost,
            evaluation=True,
        )

        def init():
            return world_model(**dummy_input, state=world_model.initial_state(None))

        return init, (world_model.__call__, world_model.loss, world_model.initial_state)

    hk_graphs = hk.multi_transform(_world_model_graphs)
    transition = hk.Transformed(init=hk_graphs.init, apply=hk_graphs.apply[0])
    loss = hk.Transformed(init=hk_graphs.init, apply=hk_graphs.apply[1])
    initial_state = hk.Transformed(init=hk_graphs.init, apply=hk_graphs.apply[2])
    return transition, loss, initial_state


def build_reverb_node(config: ModelTrainConfig, env_spec: worlds.EnvironmentSpec):
    def _build_reverb_node(env_spec: worlds.EnvironmentSpec, table_name: str) -> List[reverb.Table]:
        signature = signature = reverb_adders.SequenceAdder.signature(
            environment_spec=env_spec,
            extras_spec=worlds.ArraySpec(shape=(2,), dtype=np.int32),
            sequence_length=config.sequence_length,
        )
        rate_limiter = reverb.rate_limiters.MinSize(1)
        replay_table = reverb.Table(
            name=table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=config.replay_max_size,
            max_times_sampled=-1,
            rate_limiter=rate_limiter,
            signature=signature,
        )
        return [replay_table]

    build_reverb_node_fn = functools.partial(
        _build_reverb_node,
        env_spec=env_spec,
        table_name=config.replay_table_name,
    )
    return lp.ReverbNode(build_reverb_node_fn)


@node_utils.build_courier_node(disable_run=True)
def build_arena(config: ModelTrainConfig, game, players, reverb_handle):
    adder = reverb_adders.SequenceAdder(
        client=reverb_handle,
        priority_fns={config.replay_table_name: None},
        period=config.sequence_period or (config.sequence_length - 1),
        sequence_length=config.sequence_length,
    )
    return model_dataset_arena.Arena(game=game, players=players, reverb_adder=adder)


@node_utils.build_courier_node(disable_run=True)
def build_update_node(
    config: ModelTrainConfig,
    random_key: jax.random.PRNGKey,
    loss_graph: hk.Transformed,
    reverb_handle: lp.CourierHandle,
    result_dir: utils.ResultDirectory,
):

    format_fn = lambda x: terminal_logger_lib.data_to_string(x, "\n")

    logger = loggers.LoggerManager(
        loggers=[
            loggers.TerminalLogger(time_frequency=5, stringify_fn=format_fn),
            loggers.TensorboardLogger(result_dir.dir),
        ],
        time_frequency=5,  # Seconds.
    )
    local_counter = services.Counter()

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
    return supervised_learning.LearnerUpdate(
        loss_fn=loss_graph,
        optimizer=optimizer,
        data_iterator=data_iterator,
        logger=logger,
        counter=local_counter,
        step_key=config.step_key,
        frame_key=config.frame_key,
        random_key=random_key,
    )


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


def main(_):
    config = ModelTrainConfig()
    result_dir = utils.ResultDirectory(config.result_dir, overwrite=True)

    random_key = jax.random.PRNGKey(config.seed)
    game = build_game()
    env_spec = spec_utils.make_game_specs(game)[0]

    players = {
        0: bots.RandomIntAction(num_actions=env_spec.action.num_values),
        1: bots.RandomIntAction(num_actions=env_spec.action.num_values),
    }

    _, loss, initial_state = build_computational_graphs(config=config, env_spec=env_spec)

    program = lp.Program(name="experiment")

    with program.group("reverb"):
        reverb_handle = program.add_node(build_reverb_node(config, env_spec))

    with program.group("arena"):
        arena_handle = program.add_node(build_arena(config, game, players, reverb_handle))

    with program.group("train"):
        train_handle = program.add_node(
            build_update_node(
                config,
                random_key,
                loss,
                reverb_handle,
                result_dir.make_subdir(program._current_group),
            )
        )

    lp.launch(program, launch_type=lp.LaunchType.LOCAL_MULTI_THREADING, serialize_py_nodes=False)

    logging.info("Generating dataset...")
    arena_client = arena_handle.dereference()
    arena_client.run(num_episodes=config.replay_max_size / 100)  # Episode length is 100.
    logging.info("Dataset created.")

    train_client = train_handle.dereference()
    while True:
        train_client.step()


if __name__ == "__main__":
    app.run(main)
