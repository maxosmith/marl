"""Build IMPALA graphs."""
import dataclasses

import haiku as hk
from ml_collections import config_dict

from marl import worlds
from marl.rl.agents import impala
from marl.utils import spec_utils


@dataclasses.dataclass
class IMPALAGraphs:
    policy: hk.Transformed
    eval_policy: hk.Transformed
    loss: hk.Transformed
    initial_state: hk.Transformed
    state_spec: worlds.TreeSpec


def build_graphs(env_spec: worlds.EnvironmentSpec, config: config_dict.ConfigDict):
    timestep = spec_utils.zeros_from_spec(env_spec)

    def _impala_graphs():
        num_actions = env_spec.action.num_values
        timestep_encoder = config.timestep_encoder_ctor(num_actions=num_actions, **config.timestep_encoder_kwargs)
        memory_core = config.memory_core_ctor(**config.memory_core_kwargs)
        policy_head = config.policy_head_ctor(num_actions=num_actions, **config.policy_head_kwargs)
        value_head = config.value_head_ctor(**config.value_head_kwargs)

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
    return IMPALAGraphs(
        policy=hk.Transformed(init=hk_graphs.init, apply=hk_graphs.apply[0]),
        loss=hk.Transformed(init=hk_graphs.init, apply=hk_graphs.apply[1]),
        initial_state=hk.Transformed(init=hk_graphs.init, apply=hk_graphs.apply[2]),
        eval_policy=hk.Transformed(init=hk_graphs.init, apply=hk_graphs.apply[4]),
        state_spec=hk.without_apply_rng(hk_graphs).apply[3](None),
    )
