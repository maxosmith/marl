from typing import Optional

import haiku as hk
import jax
from absl import logging

from marl import _types, worlds
from marl.rl.replay.reverb.adders import reverb_adder
from marl.services import interfaces
from marl.utils import tree_utils


class LearnerPolicy:
    """Service providing a learner's policy/step function.

    TODO(maxsmith): Ensure that this is thread-safe.
    """

    def __init__(
        self,
        policy_fn: hk.Transformed,
        initial_state_fn: hk.Transformed,
        variable_source: interfaces.VariableSourceInterface,
        random_key: jax.random.KeyArray,
        reverb_adder: Optional[reverb_adder.ReverbAdder] = None,
        backend: Optional[str] = "cpu",
        per_episode_update: bool = False,
    ):
        """Initializes an actor.

        Args:
            policy_fn:
            initial_state_fn:
            variable_client:

        """
        logging.info(f"Initializing a learner's policy with backend {backend}.")
        self._variable_source = variable_source
        self._reverb_adder = reverb_adder
        self._per_episode_update = per_episode_update
        self._random_key = random_key
        self._policy_fn = jax.jit(policy_fn.apply, backend=backend)
        self._initial_state_fn = jax.jit(initial_state_fn.apply, backend=backend)

    def step(self, timestep: worlds.TimeStep, state: Optional[_types.Tree] = None):
        self._random_key, subkey = jax.random.split(self._random_key)
        action, new_state = self._policy_fn(self._variable_source.params, subkey, timestep, state)
        action = tree_utils.to_numpy(action)

        if not timestep.first() and self._reverb_adder:
            # Record observation and action to Reverb. Not done on the first timestep to
            # prevent double-logging this timestep.
            self._reverb_adder.add(action=action, timestep=timestep, extras=new_state)
        if not self._per_episode_update:
            # Maybe update variables with the latest copy from the updater.
            self._variable_source.update(wait=False)
        return action, new_state

    def episode_reset(self, timestep: worlds.TimeStep):
        self._random_key, subkey = jax.random.split(self._random_key)
        state = self._initial_state_fn(None, subkey, batch_size=None)

        # Start a new episode in the replay buffer.
        if not timestep.first():
            raise ValueError("Reset must be called after the first timestep.")
        if self._reverb_adder:
            self._reverb_adder.add(timestep=timestep)

        # Maybe update variables with the latest copy from the updater.
        if self._per_episode_update:
            self._variable_source.update_and_wait()

        return state

    def sync_params(self):
        """Force the learner to synchronize its parameters with its variable source."""
        self._variable_source.update_and_wait()
