"""Test for `learner_update`.

References:
 - https://github.com/google/jax/blob/main/tests/multi_device_test.py
 - https://flax.readthedocs.io/en/latest/api_reference/flax.linen/_autosummary/flax.linen.RNN.html
 - https://flax.readthedocs.io/en/latest/guides/haiku_migration_guide.html
"""
import os
from typing import Mapping, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import tree
from absl.testing import absltest, parameterized
from jax._src import xla_bridge

from marl import specs, types, worlds
from marl.services import learner_update
from marl.utils import distributed_utils, tree_utils


class _LinearRegression(nn.Module):
  """Linear regression."""

  @nn.compact
  def __call__(self, state, timestep):
    """Forward policy pass."""
    observation = jnp.asarray(timestep.observation)
    mean = self.param(
        "mean",
        lambda _, shape: jnp.ones(shape, dtype=float),  # rng, first param, is unused.
        (),
    )
    bias = self.param(
        "bias",
        lambda _, shape: jnp.ones(shape, dtype=float),  # rng, first param, is unused.
        (),
    )
    return state + 1, mean * observation + bias


class _RegressionLoss(nn.Module):

  @nn.compact
  def __call__(self, data: worlds.Trajectory, predictions: types.Tree):
    true_actions = data.action
    losses = (predictions - true_actions) ** 2
    return jnp.mean(losses), dict(losses=losses)


class _Learner(nn.Module):
  """Test policy that performs a linear operation.

  The policy computes m*x + b where the scale, m, is the parameter,
  and the bias, b, is the policy's state.
  """

  def setup(self):
    self._lin_reg = _LinearRegression()
    self._loss_fn = _RegressionLoss()

  def step(self, state, timestep):
    """Forward policy pass."""
    return self.__call__(state, timestep)

  def __call__(self, state, timestep):
    return self._lin_reg(state, timestep)

  def initialize_carry(self, rng: jax.random.PRNGKey, batch_shape: Tuple[int, ...]):
    del rng
    return jnp.ones(batch_shape[:-1], dtype=int)

  def loss(
      self,
      data: worlds.Trajectory,
  ) -> Tuple[types.Tree, Mapping[str, types.Tree]]:
    """."""
    initial_state = tree.map_structure(lambda s: s[:, 0], data.extras)

    # Define a new function to pass to jax.lax.scan
    def scan_fn(carry, timestep):
      new_carry, predictions = self._lin_reg(carry, timestep)
      return new_carry, predictions

    # Use jax.lax.scan with the newly defined function
    final_state, predictions = jax.lax.scan(scan_fn, initial_state, data)
    del final_state  # Leaving named as documentation.
    loss, loss_extras = self._loss_fn(data, predictions)
    return loss, loss_extras


_LEARNER = _Learner()
_LEARNER_INIT_PARAMS = dict(
    params=dict(
        _lin_reg=dict(
            mean=jnp.asarray(1, dtype=float), bias=jnp.asarray(1, dtype=float)
        )
    )
)


def setUpModule():
  """Run all tests with 4 spoofed CPU devices."""
  global prev_xla_flags
  prev_xla_flags = os.getenv("XLA_FLAGS")
  flags_str = prev_xla_flags or ""
  # Don't override user-specified device count, or other XLA flags.
  if "xla_force_host_platform_device_count" not in flags_str:
    os.environ["XLA_FLAGS"] = flags_str + " --xla_force_host_platform_device_count=4"
  # Clear any cached backends so new CPU backend will pick up the env var.
  xla_bridge.get_backend.cache_clear()


def tearDownModule():
  """Reset to previous device configuration in case other test modules are run."""
  if prev_xla_flags is None:
    del os.environ["XLA_FLAGS"]
  else:
    os.environ["XLA_FLAGS"] = prev_xla_flags
  xla_bridge.get_backend.cache_clear()


class LearnerUpdateTest(parameterized.TestCase):
  """Test suite for `TODO`."""

  @parameterized.parameters(
      dict(
          data=(
              worlds.Trajectory(
                  # [BATCH, TIME, ...]
                  observation=jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
                  action=jnp.array([[5, 8, 11, 14], [17, 20, 23, 26]]),  # y = 3x + 2
                  reward=jnp.array([[0, 0, 0, 0], [0, 0, 0, 0]]),
                  start_of_episode=jnp.array([[1, 0, 0, 0], [1, 0, 0, 0]]),
                  end_of_episode=jnp.array([[0, 0, 0, 1], [0, 0, 0, 1]]),
                  extras=jnp.array([[0, 0, 0, 0], [0, 0, 0, 0]]),
              ),
              worlds.Trajectory(
                  # [BATCH, TIME, ...]
                  observation=jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
                  action=jnp.array([[5, 8, 11, 14], [17, 20, 23, 26]]),  # y = 3x + 2
                  reward=jnp.array([[0, 0, 0, 0], [0, 0, 0, 0]]),
                  start_of_episode=jnp.array([[1, 0, 0, 0], [1, 0, 0, 0]]),
                  end_of_episode=jnp.array([[0, 0, 0, 1], [0, 0, 0, 1]]),
                  extras=jnp.array([[0, 0, 0, 0], [0, 0, 0, 0]]),
              ),
          ),
          step_size=1.0,
          expected_params=(
              dict(
                  params=dict(
                      _lin_reg=dict(
                          mean=jnp.asarray(-110, dtype=float),
                          bias=jnp.asarray(-19, dtype=float),
                      ),
                  ),
              ),
              dict(
                  params=dict(
                      _lin_reg=dict(
                          mean=jnp.asarray(-6062, dtype=float),
                          bias=jnp.asarray(-1078, dtype=float),
                      ),
                  ),
              ),
          ),
      ),
  )
  def test_step(
      self,
      data: Sequence[worlds.Trajectory],
      step_size: float,
      expected_params: Sequence[types.Params],
  ):
    """Tests METHOD."""
    updater = learner_update.LearnerUpdate(
        policy=_LEARNER,
        optimizer=optax.scale(step_size),
        env_spec=specs.EnvironmentSpec(
            observation=specs.ArraySpec(shape=(), dtype=float, name="observation"),
            action=specs.ArraySpec(shape=(), dtype=float, name="action"),
            reward=specs.ArraySpec(shape=(), dtype=float, name="reward"),
        ),
        random_key=jax.random.PRNGKey(42),
        data_iterator=distributed_utils.multi_device_put(
            iter(data), jax.local_devices()
        ),
    )
    tree_utils.assert_equals(updater.get_variables(), _LEARNER_INIT_PARAMS)

    for params in expected_params:
      updater.step()
      tree_utils.assert_equals(updater.get_variables(), params)

    updater.reset_training_state()
    tree_utils.assert_equals(updater.get_variables(), _LEARNER_INIT_PARAMS)


if __name__ == "__main__":
  absltest.main()
