"""Tests for train_arena."""
import numpy as np
from absl.testing import absltest, parameterized

from marl import specs, types
from marl.utils import spec_utils, tree_utils


class SpecUtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          data=(),
          spec=(),
      ),
      dict(
          data=1.0,
          spec=specs.ArraySpec((), np.float64),
      ),
      dict(
          data=(1.0, 1),
          spec=(specs.ArraySpec((), np.float64), specs.ArraySpec((), np.int64)),
      ),
      dict(
          data=(1.0, dict(x=1, y="hello")),
          spec=(
              specs.ArraySpec((), np.float64),
              dict(
                  x=specs.ArraySpec((), np.int64),
                  y=specs.ArraySpec((), np.dtype("<U5")),
              ),
          ),
      ),
  )
  def test_spec_like(self, data: types.Tree, spec: specs.TreeSpec):
    """Test cases for `spec_like`."""
    print(spec_utils.spec_like(data))
    tree_utils.assert_equals(spec, spec_utils.spec_like(data))


if __name__ == "__main__":
  absltest.main()
