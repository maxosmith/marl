"""Test cases for the `tree_utils` module."""
import numpy as np
from absl.testing import absltest, parameterized

from marl.utils import tree_utils


class TreeUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="Dict",
          input={0: 1.0, 1: 2.0},
          output={"0": 1.0, "1": 2.0},
      ),
      dict(
          testcase_name="Dict containing List",
          input={0: [1.0, 3.0], 1: 2.0},
          output={"0/0": 1.0, "0/1": 3.0, "1": 2.0},
      ),
      dict(
          testcase_name="Dict containing Arrays",
          input={0: np.array(0.0, dtype=np.float32), 1: np.array(1, dtype=np.int32)},
          output={
              "0": np.array(0.0, dtype=np.float32),
              "1": np.array(1, dtype=np.int32),
          },
      ),
  ])
  def test_flatten_as_dict(self, input, output):
    result = tree_utils.flatten_as_dict(input)
    tree_utils.assert_equals(result, output)


if __name__ == "__main__":
  absltest.main()
