"""A learning agent's policy."""
import functools
import warnings
from typing import Optional

import flax.linen as nn
import jax
import tree
from absl import logging

from marl import individuals, types, worlds
from marl.services import variable_client
from marl.utils import tree_utils


class LearnerPolicy(individuals.Agent):
  """Service providing a learner's policy/step function.

  This class is responsible for updating the policy's parameters,
  with the learning service, to be used by the data generating service.
  Thereby controlling the on/off-policiness of the learning algorithm.

  Args:
    policy
    variable_client: Client that owns the policy's parameters. This will
      typically be a `LearnerUpdate` that is learning async.
    backend: Hardware backend to run the policy on. Default is CPU.
    episode_update_freq: Frequency, in number of episodes, that the learner
      should sync their parameters. Default is after every episode (1).
    timestep_update_freq: Frequency, in number of timesteps, that the learner
      should sync their parameters.
    blocking_wait: Block on waiting for parameters to sync.
  """

  def __init__(
      self,
      policy: nn.Module,
      variable_client: variable_client.VariableClient,
      rng_key: jax.random.PRNGKey,
      backend: Optional[str] = "cpu",
      *,
      episode_update_freq: int | None = 1,
      timestep_update_freq: int | None = None,
      blocking_wait: bool = True,
  ):
    """Initializer."""
    logging.info(f"Initializing a learner's policy with backend {backend}.")
    self._variable_client = variable_client
    self._rng_key = rng_key

    if (episode_update_freq is None) and (timestep_update_freq is None):
      raise ValueError(
          "Must specify one of `episode_update_freq` or `timestep_update_freq`. "
          "If you never need to update parameters, you want a Bot implementation."
      )
    if (episode_update_freq is not None) and (timestep_update_freq is not None):
      warnings.warn(
          "Episode and timestep update frequencies are handled independently and may result "
          "in more updates than expected. Please use carefully."
      )
    self._episode_update_freq = episode_update_freq
    self._num_episodes = 0  # Tracked since last `episode` update.
    self._timestep_update_freq = timestep_update_freq
    self._num_timesteps = 0  # Tracked since last `timestep` update.
    self._blocking_wait = blocking_wait

    self._policy = policy
    self._backend = backend
    self._device = jax.local_devices(backend=self._backend)
    if len(self._device) > 1:
      warnings.warn(
          f"Found {len(self._device)} devices, but only using {self._device[0]}"
      )
    self._device = self._device[0]
    self._policy_apply = jax.jit(self._policy.apply, device=self._device)

  def step(self, timestep: worlds.TimeStep, state: Optional[types.State] = None):
    """Policy step."""
    if self._timestep_update_freq:
      self._num_timesteps += 1
      if self._num_timesteps % self._timestep_update_freq == 0:
        self._num_timesteps = 0
        logging.debug("Updating policy parameters due to step count.")
        self.update()

    action, new_state = self._policy_apply(self.params, timestep, state)
    action = tree_utils.to_numpy(action)
    return action, new_state

  def episode_reset(self, timestep: worlds.TimeStep):
    """Reset the agent's state for a new episode."""
    if not timestep.first():
      raise ValueError("Reset must be called after the first timestep.")

    if self._episode_update_freq:
      self._num_episodes += 1
      if self._num_episodes % self._episode_update_freq == 0:
        self._num_episodes = 0
        logging.debug("Updating policy parameters due to episode count.")
        self.update()

    self._rng_key, subkey = jax.random.split(self._rng_key)
    state = self._policy.initialize_carry(rng=subkey, batch_shape=())
    state = jax.device_put(state, device=self._device)
    return state

  def update(self):
    """Force the learner to synchronize its parameters with its variable source."""
    logging.debug("Synchronizing the learner's parameters.")
    self._variable_client.update(wait=self._blocking_wait)

  @property
  def params(self) -> types.Params:
    """Policy parameters."""
    return self._variable_client.params
