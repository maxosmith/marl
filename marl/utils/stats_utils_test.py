"""Test cases for the `stats_utils` module."""
import numpy as np
from absl.testing import absltest, parameterized

from marl import types
from marl.utils import stats_utils


class StatsUtilsTest(parameterized.TestCase):
  """Test cases for the `stats_utils` module."""

  @parameterized.parameters([
      dict(y=[[0.4], [0.5]], y_hat=[[0.38], [0.51]], target=0.90999),
      dict(
          y=[
              [1.6720289],
              [1.7431222],
              [0.28819785],
              [0.9109316],
              [1.2770163],
              [-0.65465194],
              [-0.9045888],
              [-1.6999624],
          ],
          y_hat=[
              [2.6915226],
              [2.738025],
              [-0.1925026],
              [1.4660996],
              [2.4726884],
              [-0.8787387],
              [-1.8647583],
              [-3.1352844],
          ],
          target=0.40119,
      ),
  ])
  def test_explained_variance(self, y: types.Array, y_hat: types.Array, target: float):
    """Basic API test for `explained_variance`."""
    output = stats_utils.explained_variance(y=np.array(y), pred=np.array(y_hat))
    self.assertAlmostEqual(output, target, places=4)


if __name__ == "__main__":
  absltest.main()
