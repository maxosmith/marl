"""JAX agent."""
import warnings
from typing import Tuple

import jax
from flax import linen as nn

from marl import individuals, types, worlds
from marl.utils import tree_utils


class JAXPolicy(nn.RNNCellBase):
  """JAX Policy."""

  @nn.module.nowrap
  def setup(self):
    """Setup the agent's variables."""

  @nn.compact
  def __call__(
      self, state: types.State, timestep: worlds.TimeStep, *args, **kwargs
  ) -> Tuple[types.State, types.Action]:
    """Forward policy pass."""

  @nn.module.nowrap
  def initialize_carry(
      self,
      rng: jax.random.PRNGKey,
      input_shape: Tuple[int, ...],
  ) -> types.State:
    """Initialize recurrent state."""


class JAXBot(individuals.Bot):
  """Bot that is implemented in JAX.

  Args:
    policy: Policy defining the bot's behavior.
    params: `policy`'s parameters.
    rng_key: Random key.
    backend: Device name to place policy graphs onto.
  """

  def __init__(
      self,
      policy: JAXPolicy,
      params: types.Params,
      rng_key: jax.random.PRNGKey,
      backend: str | None = "cpu",
  ) -> None:
    """Initializer."""
    super().__init__()
    self._policy = policy
    self._params = params
    self._rng_key = rng_key

    self._backend = backend
    self._device = jax.local_devices(backend=self._backend)
    if len(self._device) > 1:
      warnings.warn(f"Found {len(self._device)} devices, but only using {self._device[0]}")
    self._device = self._device[0]
    self._policy_apply = jax.jit(self._policy.apply, device=self._device)

  def step(self, state: types.State, timestep: worlds.TimeStep) -> Tuple[types.Action, types.State]:
    """Forward policy pass."""
    action, new_state = self._policy_apply(self._params, timestep, state)
    action = tree_utils.to_numpy(action)
    return action, new_state

  def episode_reset(self, timestep: worlds.TimeStep) -> types.State:
    """Initialize recurrent state."""
    if not timestep.first():
      raise ValueError("Reset must be called after the first timestep.")
    self._rng_key, subkey = jax.random.split(self._rng_key)
    state = self._policy.initialize_carry(rng=subkey, input_shape=())
    state = jax.device_put(state, device=self._device)
    return state
