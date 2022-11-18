"""Network modules used to build BR agents."""
from typing import Optional, Tuple

import haiku as hk
import jax.numpy as jnp

from marl import _types, nets, worlds
from marl.utils import tree_utils


class MLPTimestepEncoder(hk.Module):
    """Encodes a Timestep's info-state using an MLP."""

    def __init__(self, name: Optional[str] = "timestep_encoder"):
        """Initializes an instance of a MLPTimestepEncoder."""
        super().__init__(name=name)
        self._net = hk.nets.MLP([6, 6], activate_final=True)

    def __call__(self, timestep: worlds.TimeStep, state: _types.State) -> _types.Tree:
        """Forward pass of the encoder."""
        observation = jnp.asarray(timestep.observation["info_state"], dtype=float)
        h = self._net(observation)
        return h


class MemoryCore(hk.Module):
    def __init__(self, name: Optional[str] = "memory_core"):
        super().__init__(name=name)
        self._core = hk.LSTM(6)

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


class NoopCore(hk.Module):
    """Skip's the memory component of an IMPALA agent."""

    def __init__(self, name: Optional[str] = "memoryless_core"):
        """Initializes an instance of a NoopCore."""
        super().__init__(name=name)

    def __call__(self, inputs: _types.Tree, state: _types.State) -> Tuple[_types.Tree, _types.State]:
        """Forward pass of the core."""
        return inputs, state

    def initial_state(self, batch_size: Optional[int]) -> _types.State:
        """Gets an initial recurrent state for the core."""
        state = jnp.zeros([1])
        if batch_size is not None:
            state = tree_utils.add_batch(state, batch_size)
        return state

    def unroll(self, inputs: _types.Tree, state: nets.MLPCoreState) -> Tuple[_types.Tree, nets.MLPCoreState]:
        """This should be for additional time dimension over call"""
        return inputs, state


class PolicyHead(hk.Module):
    """Generates the policy given an agent-state."""

    def __init__(self, num_actions: int, name: Optional[str] = "policy_head"):
        """Initializes an instance of a PolicyHead."""
        super().__init__(name=name)
        self.num_actions = num_actions
        self._policy_head = hk.Linear(self.num_actions)

    def __call__(self, inputs: _types.Tree) -> Tuple[_types.Action, _types.Tree]:
        """Forward pass of the module."""
        logits = self._policy_head(inputs)  # [B, A]
        return logits


class ValueHead(hk.Module):
    """Generates a state value given an agent-state."""

    def __init__(self, num_actions: Optional[int] = None, name: Optional[str] = "value_head"):
        """Initializes an instance of a ValueHead."""
        super().__init__(name=name)
        self.num_actions = num_actions
        self._value_head = hk.Linear(1)

    def __call__(self, inputs: _types.Tree) -> Tuple[_types.Action, _types.Tree]:
        """Forward pass of the module."""
        value = self._value_head(inputs)  # [B, A].
        if not self.num_actions:
            value = jnp.squeeze(value, axis=-1)  # [B], when for state-values.
        return value
