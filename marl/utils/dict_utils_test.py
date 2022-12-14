"""Test cases for the `tree_utils` module."""
from absl.testing import absltest, parameterized

from marl.utils import dict_utils, tree_utils


class DictUtilsTest(parameterized.TestCase):
    @parameterized.parameters(
        [
            dict(
                input={0: 1.0, 1: 2.0},
                prefix="a",
                output={"a0": 1.0, "a1": 2.0},
            ),
        ]
    )
    def test_prefix_keys(self, input, prefix, output):
        result = dict_utils.prefix_keys(input, prefix)
        tree_utils.assert_equals(result, output)


if __name__ == "__main__":
    absltest.main()
