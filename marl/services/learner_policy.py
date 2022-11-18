"""A learning agent's policy."""
from typing import Optional

import haiku as hk
import jax
from absl import logging

from marl import _types, individuals, worlds
from marl.services import interfaces
from marl.services.replay.reverb.adders import reverb_adder
from marl.utils import tree_utils


class LearnerPolicy(individuals.Agent):
    """Service providing a learner's policy/step function."""

    def __init__(
        self,
        policy_fn: hk.Transformed,
        initial_state_fn: hk.Transformed,
        variable_source: interfaces.VariableSourceInterface,
        random_key: jax.random.KeyArray,
        reverb_adder: Optional[reverb_adder.ReverbAdder] = None,
        backend: Optional[str] = "cpu",
        per_episode_update: bool = True,
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
        """Policy step."""
        self._random_key, subkey = jax.random.split(self._random_key)
        action, new_state = self._policy_fn(self._variable_source.params, subkey, timestep, state)
        action = tree_utils.to_numpy(action)
        if self._reverb_adder:
            self._reverb_adder.add(timestep=timestep, action=action, extras=new_state)
        return action, new_state

    def episode_reset(self, timestep: worlds.TimeStep):
        """Reset the agent's state for a new episode."""
        self._random_key, subkey = jax.random.split(self._random_key)
        state = self._initial_state_fn(None, subkey, batch_size=None)
        # Start a new episode in the replay buffer.
        if not timestep.first():
            raise ValueError("Reset must be called after the first timestep.")
        return state

    def update(self):
        """Force the learner to synchronize its parameters with its variable source."""
        self._variable_source.update_and_wait()
