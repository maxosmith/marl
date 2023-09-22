import abc
from typing import Tuple

import jax
from flax import linen as nn

from marl import types, worlds


class JAXAgent(nn.Module, abc.ABCMeta):
  """Agent that is implemented in JAX."""

  @abc.abstractmethod
  def setup(self):
    """Setup the agent's variables."""

  def step(self, state: types.State, timestep: worlds.TimeStep) -> Tuple[types.State, types.Action]:
    """Forward policy pass."""
    return self.__call__(state, timestep)

  @abc.abstractmethod
  def __call__(self, state: types.State, timestep: worlds.TimeStep) -> Tuple[types.State, types.Action]:
    """Forward policy pass."""

  @abc.abstractclassmethod
  def initialize_carry(
      self,
      rng: jax.random.PRNGKey,
      batch_shape: Tuple[int, ...],
  ) -> types.State:
    """Initialize recurrent state."""
