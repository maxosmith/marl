"""Test for `jax_bot`."""
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from marl import types, worlds
from marl.bots import jax_bot

FAKE_TIMESTEP = worlds.TimeStep(
    step_type=worlds.StepType.FIRST,
    observation=1,
    reward=0,
)


class TestPolicyLinear(jax_bot.JAXPolicy):
  """Test policy that performs a linear operation."""

  @nn.module.nowrap
  def initialize_carry(self, rng: jax.random.PRNGKey, input_shape: Tuple[int, ...]):
    del rng
    return jnp.ones(input_shape[:-1], dtype=int)

  @nn.compact
  def __call__(
      self, timestep: worlds.TimeStep, state: types.State, *args, **kwargs
  ) -> Tuple[types.State, types.Action]:
    """Forward policy pass."""
    observation = jnp.asarray(timestep.observation)
    mean = self.param(
        "mean",
        lambda _, shape: jnp.zeros(shape, dtype=int),  # rng, first param, is unused.
        observation.shape,
    )
    return (observation * mean) + state, state + 1
