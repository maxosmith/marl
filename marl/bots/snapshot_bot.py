"""A bot backed by a snapshot."""
import pathlib
import warnings
from typing import Tuple

import jax

from marl import individuals, types, worlds
from marl.bots import jax_bot
from marl.services import snapshotter
from marl.utils import tree_utils


class SnapshotBot(individuals.Bot):
  """Parameterized-bot that caches it's parameters with a disk-backed snapshot.

  Args:
      path: Path to the snapshot.
      random_key: Random key.
      eval_policy: Whether to use the snapshot's evaluation policy or default policy.
      cache_params: Unload parameters from memory at the end of an episode.
      backend: Device name to place graphs onto.
  """

  def __init__(
      self,
      path: str | pathlib.Path,
      rng_key: jax.random.PRNGKey,
      *,
      backend: str | None = "cpu",
      cache_params: bool = True,
  ):
    """Initializer."""
    super().__init__()
    self._path = path
    self._rng_key = rng_key
    self._cache_params = cache_params

    # Set in `_maybe_load_snapshot`.
    self._snapshot = None
    self._policy = None
    self._policy_apply = None

    self._backend = backend
    self._device = None

  def step(self, state: types.State, timestep: worlds.TimeStep) -> Tuple[types.Action, types.State]:
    """Forward policy pass."""
    if isinstance(timestep.observation, dict) and ("serialized_state" in timestep.observation):
      del timestep.observation["serialized_state"]
    self._rng_key, subkey = jax.random.split(self._rng_key)
    action, new_state = self._policy_apply(self.params, state, timestep, subkey)
    action = tree_utils.to_numpy(action)
    if timestep.last():
      self._maybe_unload_snapshot()
    return action, new_state

  def episode_reset(self, timestep: worlds.TimeStep) -> types.State:
    """Initialize recurrent state."""
    if not timestep.first():
      raise ValueError("Reset must be called after the first timestep.")
    self._device = jax.local_devices(backend=self._backend)
    if len(self._device) > 1:
      warnings.warn(f"Found {len(self._device)} devices, but only using {self._device[0]}")
    self._device = self._device[0]

    self._maybe_load_snapshot()
    self._rng_key, subkey = jax.random.split(self._rng_key)
    state = self._policy.apply({}, subkey, (), method=self._policy.initialize_carry)
    state = jax.device_put(state, device=self._device)
    return state

  @property
  def params(self) -> types.Params | None:
    """Get the policy's parameters."""
    self._maybe_load_snapshot()
    return self._snapshot.params if self._snapshot else None

  def _maybe_load_snapshot(self):
    """Load the snapshot from disk."""
    if self._snapshot:
      return
    self._snapshot = snapshotter.restore_from_path(self._path)
    self._policy: jax_bot.JAXPolicy = self._snapshot.ctor(**self._snapshot.ctor_kwargs)
    self._policy_apply = jax.jit(self._policy.apply, device=self._device)

  def _maybe_unload_snapshot(self):
    """Unload the snapshot to free memory."""
    if not self._cache_params:
      return
    self._snapshot = None
