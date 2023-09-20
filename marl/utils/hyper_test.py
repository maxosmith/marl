"""Tests for marl.utils.hyper.

References:
 - https://github.com/deepmind/distribution_shift_framework/
"""
import dataclasses
from typing import Any, NamedTuple, Sequence

from absl.testing import absltest, parameterized
from ml_collections import config_dict

from marl.utils import hyper


@dataclasses.dataclass
class _ConfigDataClass:
  """Test config implemented with a dataclass."""

  a: int
  b: float
  c: bool
  e: str


class _ConfigNamedTuple(NamedTuple):
  """Test config implemented with a NamedTuple."""

  a: int
  b: float
  c: bool
  e: str


@dataclasses.dataclass
class _ConfigNestedDataClass:
  """Test config with a nested hierarchy."""

  a: int
  b: _ConfigDataClass
  c: _ConfigNamedTuple


class _ConfigNestedNamedTuple(NamedTuple):
  """Test config with a nested hierarchy."""

  a: int
  b: _ConfigDataClass
  c: _ConfigNamedTuple


class HyperTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          parameter_name="param1",
          value=1,
          expected_sweep=[
              config_dict.create(param1=1),
          ],
      ),
      dict(
          parameter_name="param2",
          value=2.5,
          expected_sweep=[
              config_dict.create(param2=2.5),
          ],
      ),
      dict(
          parameter_name="param3",
          value="test",
          expected_sweep=[
              config_dict.create(param3="test"),
          ],
      ),
      dict(
          parameter_name="param4",
          value=True,
          expected_sweep=[
              config_dict.create(param4=True),
          ],
      ),
      dict(
          parameter_name="param5",
          value=[1, 2, 3],
          expected_sweep=[
              config_dict.create(param5=[1, 2, 3]),
          ],
      ),
      dict(
          parameter_name="param6",
          value={"key1": "value1"},
          expected_sweep=[
              config_dict.create(param6={"key1": "value1"}),
          ],
      ),
      dict(
          parameter_name="param7",
          value=config_dict.create(key1="value1"),
          expected_sweep=[
              config_dict.create(param7=config_dict.create(key1="value1")),
          ],
      ),
  )
  def test_fixed(self, parameter_name: str, value: Sequence[Any], expected_sweep: hyper.Sweep):
    """Test the `fixed` function."""
    self.assertEqual(expected_sweep, hyper.fixed(parameter_name, value))

  @parameterized.parameters([
      dict(
          parameter_name="a",
          values=[1, 2, 3],
          expected_sweep=[
              config_dict.create(a=1),
              config_dict.create(a=2),
              config_dict.create(a=3),
          ],
      ),
      dict(
          parameter_name="b",
          values=[0.1, 0.2, 0.3],
          expected_sweep=[
              config_dict.create(b=0.1),
              config_dict.create(b=0.2),
              config_dict.create(b=0.3),
          ],
      ),
      dict(
          parameter_name="c",
          values=[True, False],
          expected_sweep=[
              config_dict.create(c=True),
              config_dict.create(c=False),
          ],
      ),
      dict(
          parameter_name="d",
          values=["one", "two", "three"],
          expected_sweep=[
              config_dict.create(d="one"),
              config_dict.create(d="two"),
              config_dict.create(d="three"),
          ],
      ),
      dict(
          parameter_name="e",
          values=[1, 0.5, True, "string"],
          expected_sweep=[
              config_dict.create(e=1),
              config_dict.create(e=0.5),
              config_dict.create(e=True),
              config_dict.create(e="string"),
          ],
      ),
      dict(parameter_name="f", values=[], expected_sweep=[]),
  ])
  def test_sweep(self, parameter_name: str, values: Sequence[Any], expected_sweep: hyper.Sweep):
    """Test the `sweep` function."""
    self.assertEqual(expected_sweep, hyper.sweep(parameter_name, values))

  @parameterized.parameters([
      dict(sweeps=[], expected_sweep=[config_dict.ConfigDict()]),
      dict(
          sweeps=[hyper.sweep("param1", [1, 2, 3, 4, 5, 6])],
          expected_sweep=[
              config_dict.create(param1=1),
              config_dict.create(param1=2),
              config_dict.create(param1=3),
              config_dict.create(param1=4),
              config_dict.create(param1=5),
              config_dict.create(param1=6),
          ],
      ),
      dict(
          sweeps=[hyper.sweep("param1", [1, 2, 3]), hyper.sweep("param2", [4, 5, 6])],
          expected_sweep=[
              config_dict.create(param1=1, param2=4),
              config_dict.create(param1=1, param2=5),
              config_dict.create(param1=1, param2=6),
              config_dict.create(param1=2, param2=4),
              config_dict.create(param1=2, param2=5),
              config_dict.create(param1=2, param2=6),
              config_dict.create(param1=3, param2=4),
              config_dict.create(param1=3, param2=5),
              config_dict.create(param1=3, param2=6),
          ],
      ),
      dict(
          sweeps=[
              hyper.sweep("param1", [1, 2]),
              hyper.sweep("param2", [3, 4]),
              hyper.sweep("param3", [5, 6]),
          ],
          expected_sweep=[
              config_dict.create(param1=1, param2=3, param3=5),
              config_dict.create(param1=1, param2=3, param3=6),
              config_dict.create(param1=1, param2=4, param3=5),
              config_dict.create(param1=1, param2=4, param3=6),
              config_dict.create(param1=2, param2=3, param3=5),
              config_dict.create(param1=2, param2=3, param3=6),
              config_dict.create(param1=2, param2=4, param3=5),
              config_dict.create(param1=2, param2=4, param3=6),
          ],
      ),
      dict(
          sweeps=[
              hyper.sweep("param1", [1, 2.0, "Three"]),
              hyper.sweep("param2", [True, "Two", 3.0]),
          ],
          expected_sweep=[
              config_dict.create(param1=1, param2=True),
              config_dict.create(param1=1, param2="Two"),
              config_dict.create(param1=1, param2=3.0),
              config_dict.create(param1=2.0, param2=True),
              config_dict.create(param1=2.0, param2="Two"),
              config_dict.create(param1=2.0, param2=3.0),
              config_dict.create(param1="Three", param2=True),
              config_dict.create(param1="Three", param2="Two"),
              config_dict.create(param1="Three", param2=3.0),
          ],
      ),
  ])
  def test_product(self, sweeps: Sequence[hyper.Sweep], expected_sweep: hyper.Sweep):
    """Test the `product` function."""
    self.assertEqual(expected_sweep, hyper.product(sweeps))

  @parameterized.parameters([
      dict(sweeps=[], expected_sweep=[]),
      dict(
          sweeps=[hyper.sweep("param1", [1, 2, 3, 4, 5, 6])],
          expected_sweep=[
              config_dict.create(param1=1),
              config_dict.create(param1=2),
              config_dict.create(param1=3),
              config_dict.create(param1=4),
              config_dict.create(param1=5),
              config_dict.create(param1=6),
          ],
      ),
      dict(
          sweeps=[hyper.sweep("param1", [1, 2, 3]), hyper.sweep("param2", [4, 5, 6])],
          expected_sweep=[
              config_dict.create(param1=1, param2=4),
              config_dict.create(param1=2, param2=5),
              config_dict.create(param1=3, param2=6),
          ],
      ),
      dict(
          sweeps=[
              hyper.sweep("param1", [1, 2, 3]),
              hyper.sweep("param2", [4, 5, 6]),
              hyper.sweep("param3", [7, 8, 9]),
          ],
          expected_sweep=[
              config_dict.create(param1=1, param2=4, param3=7),
              config_dict.create(param1=2, param2=5, param3=8),
              config_dict.create(param1=3, param2=6, param3=9),
          ],
      ),
      dict(
          sweeps=[
              hyper.sweep("param1", [1, 2.0, "Three"]),
              hyper.sweep("param2", [True, "Two", 3.0]),
          ],
          expected_sweep=[
              config_dict.create(param1=1, param2=True),
              config_dict.create(param1=2.0, param2="Two"),
              config_dict.create(param1="Three", param2=3.0),
          ],
      ),
      dict(
          sweeps=[
              hyper.sweep("param1", [1, 2, 3]),
              hyper.sweep("param2", [4, 5, 6, 7]),
          ],
          expected_sweep=[
              config_dict.create(param1=1, param2=4),
              config_dict.create(param1=2, param2=5),
              config_dict.create(param1=3, param2=6),
          ],
      ),
  ])
  def test_zipit(self, sweeps, expected_sweep):
    """Test the `zipit` function."""
    self.assertEqual(expected_sweep, hyper.zipit(sweeps))

  @parameterized.parameters(
      # Test case 1: Basic functionality with non-overlapping keys.
      dict(
          base=config_dict.create(param1=1, param2=2),
          overrides=[config_dict.create(param3=3), config_dict.create(param4=4)],
          expected_sweep=[
              config_dict.create(param1=1, param2=2, param3=3),
              config_dict.create(param1=1, param2=2, param4=4),
          ],
      ),
      # Test case 2: Overrides contain overlapping keys with base.
      dict(
          base=config_dict.create(param1=1, param2=2),
          overrides=[
              config_dict.create(param1=2, param2=3),
              config_dict.create(param1=3, param2=4),
          ],
          expected_sweep=[
              config_dict.create(param1=2, param2=3),
              config_dict.create(param1=3, param2=4),
          ],
      ),
      # Test case 3: Base and overrides contain overlapping keys, with nested dictionaries.
      dict(
          base=config_dict.create(param1=config_dict.create(subparam1=1), param2=2),
          overrides=[
              config_dict.create(param1=config_dict.create(subparam1=2)),
              config_dict.create(param1=config_dict.create(subparam1=3, subparam2=3)),
          ],
          expected_sweep=[
              config_dict.create(param1=config_dict.create(subparam1=2), param2=2),
              config_dict.create(param1=config_dict.create(subparam1=3, subparam2=3), param2=2),
          ],
      ),
      # Test case 4: Overrides contain keys not in base.
      dict(
          base=config_dict.create(param1=1, param2=2),
          overrides=[config_dict.create(param3=3), config_dict.create(param4=4)],
          expected_sweep=[
              config_dict.create(param1=1, param2=2, param3=3),
              config_dict.create(param1=1, param2=2, param4=4),
          ],
      ),
      # Test case 5: Base is an empty dictionary.
      dict(
          base=config_dict.create(),
          overrides=[config_dict.create(param1=1), config_dict.create(param2=2)],
          expected_sweep=[
              config_dict.create(param1=1),
              config_dict.create(param2=2),
          ],
      ),
      # Test case 6: Overrides is an empty list.
      dict(
          base=config_dict.create(param1=1, param2=2),
          overrides=[],
          expected_sweep=[],
      ),
      # Test case 7: Both base and overrides are empty.
      dict(
          base=config_dict.create(),
          overrides=[],
          expected_sweep=[],
      ),
  )
  def test_default(
      self,
      base: config_dict,
      overrides: Sequence[hyper.Sweep],
      expected_sweep: hyper.Sweep,
  ):
    """Test the `default` function."""
    self.assertEqual(expected_sweep, hyper.default(base, overrides))

  @parameterized.parameters(
      dict(
          config_type=_ConfigDataClass,
          sweep=[config_dict.create(a=1, b=2.0, c=True, e="x")],
          expected_casts=[_ConfigDataClass(a=1, b=2.0, c=True, e="x")],
      ),
      dict(
          config_type=_ConfigDataClass,
          sweep=[
              config_dict.create(a=1, b=2.0, c=True, e="x"),
              config_dict.create(a=2, b=3.14, c=False, e="y"),
          ],
          expected_casts=[
              _ConfigDataClass(a=1, b=2.0, c=True, e="x"),
              _ConfigDataClass(a=2, b=3.14, c=False, e="y"),
          ],
      ),
      dict(
          config_type=_ConfigNamedTuple,
          sweep=[config_dict.create(a=1, b=2.0, c=True, e="x")],
          expected_casts=[_ConfigNamedTuple(a=1, b=2.0, c=True, e="x")],
      ),
      dict(
          config_type=_ConfigNestedDataClass,
          sweep=[
              config_dict.create(
                  a=1,
                  b=config_dict.create(a=1, b=2.0, c=True, e="x"),
                  c=config_dict.create(a=1, b=2.0, c=True, e="x"),
              )
          ],
          expected_casts=[
              _ConfigNestedDataClass(
                  a=1,
                  b=_ConfigDataClass(a=1, b=2.0, c=True, e="x"),
                  c=_ConfigNamedTuple(a=1, b=2.0, c=True, e="x"),
              )
          ],
      ),
      dict(
          config_type=_ConfigNestedNamedTuple,
          sweep=[
              config_dict.create(
                  a=1,
                  b=config_dict.create(a=1, b=2.0, c=True, e="x"),
                  c=config_dict.create(a=1, b=2.0, c=True, e="x"),
              )
          ],
          expected_casts=[
              _ConfigNestedNamedTuple(
                  a=1,
                  b=_ConfigDataClass(a=1, b=2.0, c=True, e="x"),
                  c=_ConfigNamedTuple(a=1, b=2.0, c=True, e="x"),
              )
          ],
      ),
  )
  def test_cast(
      self,
      config_type: Any,
      sweep: Sequence[hyper.Sweep],
      expected_casts: Sequence[Any],
  ):
    """Test the `cast` function."""
    casted_sweep = hyper.cast(config_type, sweep)
    self.assertListEqual(casted_sweep, expected_casts)


if __name__ == "__main__":
  absltest.main()
