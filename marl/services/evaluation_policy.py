from typing import Optional

import haiku as hk
import jax
from absl import logging

from marl import _types, worlds
from marl.rl.replay.reverb.adders import reverb_adder
from marl.services import interfaces
from marl.utils import tree_utils


class EvaluationPolicy:
    """Service providing a policy/step function.

    TODO(maxsmith): Ensure that this is thread-safe.
    """

    def __init__(
        self,
        policy_fn: hk.Transformed,
        initial_state_fn: hk.Transformed,
        variable_source: interfaces.VariableSourceInterface,
        random_key: jax.random.KeyArray,
        backend: Optional[str] = "cpu",
    ):
        """Initializes an actor.

        Args:
            policy_fn:
            initial_state_fn:
            variable_client:

        """
        logging.info(f"Initializing an evaluation policy with backend {backend}.")
        self._variable_source = variable_source
        self._reverb_adder = reverb_adder
        self._random_key = random_key
        self._policy_fn = jax.jit(policy_fn.apply, backend=backend)
        self._initial_state_fn = jax.jit(initial_state_fn.apply, backend=backend)

    def step(self, timestep: worlds.TimeStep, state: Optional[_types.Tree] = None):
        if timestep.first():
            state = self.episode_reset(timestep)
        self._random_key, subkey = jax.random.split(self._random_key)
        action, new_state = self._policy_fn(self._variable_source.params, subkey, timestep, state)
        action = tree_utils.to_numpy(action)
        return action, new_state

    def episode_reset(self, timestep: worlds.TimeStep):
        self._random_key, subkey = jax.random.split(self._random_key)
        state = self._initial_state_fn(None, subkey, batch_size=None)
        if not timestep.first():
            raise ValueError("Reset must be called after the first timestep.")
        return state

    def sync_params(self):
        """Force the learner to synchronize its parameters with its variable source."""
        self._variable_source.update_and_wait()
