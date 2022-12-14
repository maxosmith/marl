from typing import Optional, Tuple

import haiku as hk
import jax.numpy as jnp
import numpy as np

from marl import _types, worlds
from marl.agents.impala.impala import IMPALAState


class TimestepEncoder(hk.Module):
    def __init__(self, name: Optional[str] = "timestep_encoder"):
        super().__init__(name=name)
        self._net = hk.nets.MLP([3, 3])

    def __call__(self, timestep: worlds.TimeStep, state: IMPALAState) -> _types.Tree:
        del state  # Could be used for previous action or other engineered observations.
        return self._net(timestep.observation.astype(float))


class MemoryCore(hk.Module):
    def __init__(self, name: Optional[str] = "memory_core"):
        super().__init__(name=name)
        self._core = hk.LSTM(3)

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
        self._policy_layer = hk.Linear(num_actions)

    def __call__(self, inputs: _types.Tree) -> Tuple[_types.Action, _types.Tree]:
        logits = self._policy_layer(inputs)  # [B, A]
        return logits


class ValueHead(hk.Module):
    def __init__(self, name: Optional[str] = "value_head"):
        super().__init__(name=name)
        self._value_layer = hk.Linear(1)

    def __call__(self, inputs: _types.Tree) -> Tuple[_types.Action, _types.Tree]:
        value = jnp.squeeze(self._value_layer(inputs), axis=-1)  # [B]
        return value
