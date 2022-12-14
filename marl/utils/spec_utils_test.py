import numpy as np
from absl.testing import absltest, parameterized

from marl import worlds
from marl.utils import spec_utils


class SpecUtilsTest(parameterized.TestCase):
    @parameterized.named_parameters(
        [
            dict(
                testcase_name="Array",
                data=np.ones([3], dtype=int),
                data_spec=worlds.ArraySpec(shape=[3], dtype=int),
            ),
            dict(
                testcase_name="Tensor",
                data=np.ones([3, 4, 5], dtype=int),
                data_spec=worlds.ArraySpec(shape=[3, 4, 5], dtype=int),
            ),
            dict(
                testcase_name="Heterogeneous List",
                data=[np.ones([3], dtype=int), np.ones([3, 4, 5], dtype=float)],
                data_spec=[worlds.ArraySpec(shape=[3], dtype=int), worlds.ArraySpec(shape=[3, 4, 5], dtype=float)],
            ),
            dict(
                testcase_name="Homogeneous List",
                data=[np.ones([3, 4, 5], dtype=float), np.ones([3, 4, 5], dtype=float)],
                data_spec=[worlds.ArraySpec(shape=[3, 4, 5], dtype=float), worlds.ArraySpec(shape=[3, 4, 5], dtype=float)],
            ),
            dict(
                testcase_name="Dictionary",
                data=dict(x=np.ones([3], dtype=int), y=np.ones([3], dtype=float)),
                data_spec=dict(x=worlds.ArraySpec(shape=[3], dtype=int), y=worlds.ArraySpec(shape=[3], dtype=float)),
            ),
            dict(
                testcase_name="Tree",
                data=dict(x=np.ones([3], dtype=int), y=[np.ones([3], dtype=float), np.ones([3], dtype=float)]),
                data_spec=dict(
                    x=worlds.ArraySpec(shape=[3], dtype=int),
                    y=[worlds.ArraySpec(shape=[3], dtype=float), worlds.ArraySpec(shape=[3], dtype=float)],
                ),
            ),
        ]
    )
    def test_make_tree_spec(self, data, data_spec):
        result = spec_utils.make_tree_spec(data)
        spec_utils.assert_equal_tree_specs(result, data_spec)


if __name__ == "__main__":
    absltest.main()
