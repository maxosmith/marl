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
from marl.utils import import_utils, signals, spec_utils, wrappers
from marl_experiments.gathering import networks
from marl_experiments.gathering import world_model as world_model_lib


@dataclasses.dataclass
class IMPALAGraphs:
    policy: hk.Transformed
    eval_policy: hk.Transformed
    loss: hk.Transformed
    initial_state: hk.Transformed
    state_spec: worlds.TreeSpec


def build_agent_graphs(config, env_spec: worlds.EnvironmentSpec):
    timestep = spec_utils.zeros_from_spec(env_spec)

    def _impala_graphs():
        num_actions = env_spec.action.num_values

        timestep_encoder = import_utils.initialize(
            config.timestep_encoder_ctor, num_actions=num_actions, **config.timestep_encoder_kwargs
        )
        memory_core = import_utils.initialize(config.memory_core_ctor, **config.memory_core_kwargs)
        policy_head = import_utils.initialize(
            config.policy_head_ctor, num_actions=num_actions, **config.policy_head_kwargs
        )
        value_head = import_utils.initialize(config.value_head_ctor, **config.value_head_kwargs)

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
    return IMPALAGraphs(
        policy=train_policy, loss=loss, initial_state=initial_state, state_spec=state_spec, eval_policy=eval_policy
    )


@dataclasses.dataclass
class WorldGraphs:
    transition: hk.Transformed
    loss: hk.Transformed
    initial_state: hk.Transformed


def build_world_graphs(config: config_dict.ConfigDict, env_spec: worlds.EnvironmentSpec):
    """Build world-model graphs."""
    dummy_input = dict(
        world_state={0: spec_utils.zeros_from_spec(env_spec), 1: spec_utils.zeros_from_spec(env_spec)},
        actions={0: np.zeros((), dtype=np.int32), 1: np.zeros((), dtype=np.int32)},
    )

    def _world_model_graphs():
        """Helper function for building world graphs via closure."""
        memory_core = import_utils.initialize(config.memory_core_ctor, **config.memory_core_kwargs)

        world_model = world_model_lib.WorldModel(
            num_players=2,
            state_shape=env_spec.observation.shape,
            input_encoder=networks.WorldStateLinearEncoder(
                state_shape=env_spec.observation.shape, num_actions=env_spec.action.num_values
            ),
            memory_core=memory_core,
            observation_prediction_head=networks.WorldStateLinearPredictionHead(state_shape=env_spec.observation.shape),
            reward_prediction_head=networks.RewardPredictionHead(),
            reward_cost=config.reward_cost,
            evaluation=True,
        )

        def init():
            """Initialize all model parameters, assuming they're all called used in the call."""
            return world_model(**dummy_input, memory=world_model.initial_state(None))

        return init, (world_model.__call__, world_model.loss, world_model.initial_state)

    hk_graphs = hk.multi_transform(_world_model_graphs)
    transition = hk.Transformed(init=hk_graphs.init, apply=hk_graphs.apply[0])
    loss = hk.Transformed(init=hk_graphs.init, apply=hk_graphs.apply[1])
    initial_state = hk.Transformed(init=hk_graphs.init, apply=hk_graphs.apply[2])
    return WorldGraphs(transition=transition, loss=loss, initial_state=initial_state)
