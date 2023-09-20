"""Test suite for `array_utils`."""
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from marl.utils import array_utils


class YourModuleNameTest(parameterized.TestCase):
  """Test suite for `array_utils`."""

  @parameterized.parameters(
      (0, 0),
      (1.1, 0.0),
      (np.array([1, 2]), np.array([0, 0])),
      (jnp.array([1, 2]), jnp.array([0, 0])),
      (np.int32(3), np.int32(0)),
      (np.float64(4.4), np.float64(0.0)),
  )
  def test_zeros_like(self, input_value, expected_output):
    """Tests `zeros_like` function."""
    result = array_utils.zeros_like(input_value)
    if isinstance(input_value, jax.Array):
      self.assertTrue(jnp.array_equal(result, expected_output))
    else:
      np.testing.assert_equal(result, expected_output)

  @parameterized.parameters(
      ([0], 1, [[1]]),
      ([0], 2, [[1, 0]]),
      ([0, 1, 2], 3, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
      ([1, 2, 0], 3, [[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
  )
  def test_one_hot(self, input_array, num_classes, expected_output):
    """Tests `one_hot` function."""
    for op_lib in [np, jnp]:
      input_array = op_lib.asarray(input_array)
      expected_output = op_lib.asarray(expected_output)
      result = array_utils.one_hot(input_array, num_classes)
      self.assertTrue(op_lib.array_equal(result, expected_output))

  @parameterized.parameters(
      ([1, 2], 3, [1, 2, 3]),
      ([[1, 2], [4, 5]], 3, [[1, 2, 3], [4, 5, 3]]),
      ([[1, 2], [4, 5]], [3], [[1, 2, 3], [4, 5, 3]]),
      ([[1, 2], [4, 5]], [3, 7], [[1, 2, 3, 7], [4, 5, 3, 7]]),
  )
  def test_broadcast_concat(self, x, y, expected_output):
    """Tests `broadcast_concat` function."""
    for op_lib in [np, jnp]:
      x = op_lib.asarray(x)
      y = op_lib.asarray(y)
      expected_output = op_lib.asarray(expected_output)
      result = array_utils.broadcast_concat(x, y)
      self.assertTrue(op_lib.array_equal(result, expected_output))


if __name__ == "__main__":
  absltest.main()
