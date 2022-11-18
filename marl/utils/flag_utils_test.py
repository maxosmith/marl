"""Test cases for the `flag_utils` module."""
from typing import Sequence

from absl.testing import absltest, parameterized
from ml_collections import config_dict

from marl.utils import flag_utils


class FlagUtilsTest(parameterized.TestCase):
    """Test cases for the `flat_utils` module."""

    @parameterized.parameters(
        [
            dict(
                overrides=["x = 1"],
                base_config=config_dict.create(x=0),
                result_config=config_dict.create(x=1),
            ),
        ]
    )
    def test_apply_overrides(
        self, overrides: Sequence[str], base_config: config_dict.ConfigDict, result_config: config_dict.ConfigDict
    ):
        """Basic API test for `apply_overrides`."""
        output = flag_utils.apply_overrides(overrides, base_config)
        for key, value in result_config.items():
            self.assertEqual(output[key], value)


if __name__ == "__main__":
    absltest.main()
