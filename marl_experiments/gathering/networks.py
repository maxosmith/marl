from typing import Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from marl import _types, worlds
from marl.rl.agents.impala.graphs import IMPALAState


class TimestepEncoder(hk.Module):
    def __init__(self, num_actions: int, name: Optional[str] = "timestep_encoder"):
        super().__init__(name=name)
        self.num_actions = num_actions
        self._observation_net = hk.Sequential(
            [
                # Input: [h, w, 4].
                hk.ConvND(num_spatial_dims=2, output_channels=2, kernel_shape=2, stride=1),
                jax.nn.relu,
                hk.ConvND(num_spatial_dims=2, output_channels=1, kernel_shape=2, stride=1),
                jax.nn.relu,
                hk.Flatten(),
            ]
        )
        self._net = hk.nets.MLP([20, 20])

    def __call__(self, timestep: worlds.TimeStep, state: IMPALAState) -> _types.Tree:
        observation = timestep.observation.astype(float)
        # ConvND assumes there is a leading batch dimension: [B, H, W, C].
        if len(observation.shape) == 3:
            # Add a dummy batch dimension, this typically needs to happen during graph building.
            observation = jnp.expand_dims(observation, axis=0)
            h = self._observation_net(observation)
            # Remove the dummy batch dimension.
            h = jnp.squeeze(h, axis=0)
        else:
            h = self._observation_net(observation)
        action = jax.nn.one_hot(state.prev_action, self.num_actions)
        h = jnp.concatenate([h, action], axis=-1)
        return self._net(h)


class MemoryCore(hk.Module):
    def __init__(self, name: Optional[str] = "memory_core"):
        super().__init__(name=name)
        self._core = hk.LSTM(20)

    def __call__(self, inputs: _types.Tree, state: hk.LSTMState) -> Tuple[_types.Tree, hk.LSTMState]:
        outputs, new_state = self._core(inputs, state)
        return outputs, new_state

    def initial_state(self, batch_size: Optional[int]) -> hk.LSTMState:
        return self._core.initial_state(batch_size)

    def unroll(self, inputs: _types.Tree, state: hk.LSTMState) -> Tuple[_types.Tree, hk.LSTMState]:
        """This should be for additional time dimension over call"""
        outputs, new_state = hk.static_unroll(
            core=self._core, input_sequence=inputs, initial_state=state, time_major=False
        )
        return outputs, new_state


class PolicyHead(hk.Module):
    def __init__(self, num_actions: int, name: Optional[str] = "policy_head"):
        super().__init__(name=name)
        self.num_actions = num_actions
        self._policy_head = hk.nets.MLP([10, self.num_actions])

    def __call__(self, inputs: _types.Tree) -> Tuple[_types.Action, _types.Tree]:
        logits = self._policy_head(inputs)  # [B, A]
        return logits


class ValueHead(hk.Module):
    def __init__(self, name: Optional[str] = "value_head"):
        super().__init__(name=name)
        self._value_head = hk.nets.MLP([10, 1])

    def __call__(self, inputs: _types.Tree) -> Tuple[_types.Action, _types.Tree]:
        value = jnp.squeeze(self._value_head(inputs), axis=-1)  # [B]
        return value
