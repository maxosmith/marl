"""Policy that follows a demonstrator.

TODO(maxsmith): This is super hacky for debugging IMPALA, refactor.
"""
from typing import Any, Optional

import haiku as hk
import jax
import numpy as np
from absl import logging

from marl import _types, worlds
from marl.rl.replay.reverb.adders import reverb_adder
from marl.services import interfaces
from marl.utils import tree_utils


class DemonstrationPolicy:
    """Service providing a learner's policy/step function."""

    def __init__(
        self,
        demonstrator: Any,
        reverb_adder: reverb_adder.ReverbAdder,
        initial_state_fn: hk.Transformed,
        random_key: jax.random.KeyArray,
    ):
        """Initializes an actor.

        Args:
            policy_fn:
            initial_state_fn:
            variable_client:

        """
        self._demonstrator = demonstrator
        self._reverb_adder = reverb_adder
        self._initial_state_fn = initial_state_fn
        self._random_key = random_key
        self._initial_state_fn = jax.jit(initial_state_fn.apply, backend="cpu")

    def step(self, timestep: worlds.TimeStep, state: Optional[_types.Tree] = None):
        action, _ = self._demonstrator.step(timestep, state)
        action = action.astype(np.int32)

        # Update the fake state following `IMPALA`.
        logits = np.zeros_like(state.logits)
        logits[int(action)] = 1
        state = state._replace(logits=logits, prev_action=action)

        if not timestep.first():
            # Record observation and action to Reverb. Not done on the first timestep to
            # prevent double-logging this timestep.
            self._reverb_adder.add(action=action, timestep=timestep, extras=state)
        return action, state

    def episode_reset(self, timestep: worlds.TimeStep):
        self._random_key, subkey = jax.random.split(self._random_key)
        state = self._initial_state_fn(None, subkey, batch_size=None)
        # Start a new episode in the replay buffer.
        if not timestep.first():
            raise ValueError("Reset must be called after the first timestep.")
        self._reverb_adder.add(timestep=timestep)
        return state
