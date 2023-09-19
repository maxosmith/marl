"""Test for `TODO`."""
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from marl.utils import schedules


class SchedulesTest(parameterized.TestCase):
  """Test suite for `marl.utils.schedules`."""

  @parameterized.parameters((0.5, 0.0, 0.5), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0))
  def test_constant(self, x: float, t: float, expected: float):
    """Tests Constant schedule."""
    constant = schedules.Constant(x)
    self.assertEqual(constant(jnp.array(t)), jnp.array(expected))

  @parameterized.parameters((0.0, 1.0, 10, 0, 0.0), (0.0, 1.0, 10, 5, 0.5), (0.0, 1.0, 10, 10, 1.0))
  def test_linear(self, x_initial: float, x_final: float, num_steps: int, t: int, expected: float):
    """Tests Linear schedule."""
    linear = schedules.Linear(x_initial, x_final, num_steps)
    self.assertEqual(linear(jnp.array(t)), jnp.array(expected))

  @parameterized.parameters((1.0, 5, 0.0, 0, 0.0), (1.0, 5, 0.0, 5, 1.0), (1.0, 5, 0.0, 10, 1.0))
  def test_step(self, x_final: float, num_steps: int, x_initial: float, t: int, expected: float):
    """Tests Step schedule."""
    step = schedules.Step(x_final, num_steps, x_initial)
    self.assertEqual(step(jnp.array(t)), jnp.array(expected))

  @parameterized.parameters(
      (0.0, 1.0, 5, 10, 0, 0.0),
      (0.0, 1.0, 5, 10, 5, 0.0),
      (0.0, 1.0, 5, 10, 7, 0.4),
      (0.0, 1.0, 5, 10, 10, 1.0),
  )
  def test_rectified_linear(
      self,
      x_initial: float,
      x_final: float,
      num_steps_start: int,
      num_steps_end: int,
      t: int,
      expected: float,
  ):
    """Tests RectifiedLinear schedule."""
    rectified_linear = schedules.RectifiedLinear(x_initial, x_final, num_steps_start, num_steps_end)
    self.assertEqual(rectified_linear(jnp.array(t)), jnp.array(expected))


if __name__ == "__main__":
  absltest.main()
