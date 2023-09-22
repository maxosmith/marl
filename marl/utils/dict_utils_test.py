"""Test for Dictionary utility operations."""
from typing import Callable, Mapping, TypeVar

from absl.testing import absltest, parameterized

from marl.utils import dict_utils

_Key = TypeVar("_Key")
_ModKey = TypeVar("_ModKey")
_Value = TypeVar("_Value")
_ModValue = TypeVar("_ModValue")


class DictionaryUtilityTest(parameterized.TestCase):
  """Test suite for Dictionary Utility Operations."""

  @parameterized.parameters(
      ({"a": 1, "b": 2}, lambda x: x.upper(), {"A": 1, "B": 2}),
      ({1: "a", 2: "b"}, lambda x: x * 2, {2: "a", 4: "b"}),
  )
  def test_key_apply(
      self,
      input_dict: Mapping[_Key, _Value],
      func: Callable[[_Key], _ModKey],
      expected_output: Mapping[_ModKey, _Key],
  ):
    """Tests key_apply function."""
    self.assertEqual(dict_utils.key_apply(input_dict, func), expected_output)

  @parameterized.parameters(
      ({"a": 1, "b": 2}, lambda x: x * 2, {"a": 2, "b": 4}),
      ({"a": "hello", "b": "world"}, lambda x: x.upper(), {"a": "HELLO", "b": "WORLD"}),
  )
  def test_value_apply(
      self,
      input_dict: Mapping[_Key, _Value],
      func: Callable[[_Value], _ModValue],
      expected_output: Mapping[_Key, _ModValue],
  ):
    """Tests value_apply function."""
    self.assertEqual(dict_utils.value_apply(input_dict, func), expected_output)

  @parameterized.parameters(
      ({"a": 1, "b": 2}, "prefix", "/", {"prefix/a": 1, "prefix/b": 2}),
      ({"x": 3, "y": 4}, "pre", "_", {"pre_x": 3, "pre_y": 4}),
  )
  def test_prefix_keys(
      self,
      input_dict: Mapping[str, _Value],
      prefix: str,
      delimiter: str,
      expected_output: Mapping[str, _Value],
  ):
    """Tests prefix_keys function."""
    self.assertEqual(dict_utils.prefix_keys(input_dict, prefix, delimiter), expected_output)

  @parameterized.parameters(
      ({"prefix/a": 1, "prefix/b": 2}, "prefix", "/", {"a": 1, "b": 2}),
      ({"pre_x": 3, "pre_y": 4}, "pre", "_", {"x": 3, "y": 4}),
  )
  def test_unprefix_keys(
      self,
      input_dict: Mapping[str, _Value],
      prefix: str,
      delimiter: str,
      expected_output: Mapping[str, _Value],
  ):
    """Tests unprefix_keys function."""
    self.assertEqual(dict_utils.unprefix_keys(input_dict, prefix, delimiter), expected_output)


if __name__ == "__main__":
  absltest.main()
