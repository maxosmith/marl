"""Test cases for the `dict_utils` module."""
from absl.testing import absltest, parameterized

from marl.utils import dict_utils, tree_utils


class DictUtilsTest(parameterized.TestCase):
    """Test cases for dictionary utility functions."""

    @parameterized.parameters(
        [
            dict(
                input={0: 1.0, 1: 2.0},
                prefix="a",
                output={"a/0": 1.0, "a/1": 2.0},
            ),
        ]
    )
    def test_prefix_keys(self, input, prefix, output):
        """Test prefixing keys."""
        result = dict_utils.prefix_keys(input, prefix)
        tree_utils.assert_equals(result, output)

    @parameterized.parameters(
        [
            dict(
                input={"a/0": 1.0, "a/1": 2.0},
                prefix="a",
                output={"0": 1.0, "1": 2.0},
            ),
        ]
    )
    def test_unprefix_keys(self, input, prefix, output):
        """Test removing prefix from keys."""
        result = dict_utils.unprefix_keys(input, prefix)
        tree_utils.assert_equals(result, output)


if __name__ == "__main__":
    absltest.main()
