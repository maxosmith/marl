"""Dictionary utility operations."""
from typing import Callable, Mapping, TypeVar

_Key = TypeVar("_Key")
_ModKey = TypeVar("_ModKey")
_Value = TypeVar("_Value")
_ModValue = TypeVar("_ModValue")


def key_apply(x: Mapping[_Key, _Value], fn: Callable[[_Key], _ModKey]) -> Mapping[_ModKey, _Value]:
  """Apply function over keys.

  Args:
    x: Mapping to modify.
    fn: Function to apply over the mapping's keys.

  Returns:
    Collection with keys modified following `fn`.
  """
  return {fn(k): v for k, v in x.items()}


def value_apply(x: Mapping[_Key, _Value], fn: Callable[[_Value], _ModValue]) -> Mapping[_Key, _ModValue]:
  """Apply function over values.

  Args:
    x: Mapping to modify.
    fn: Function to apply over the mapping's values.

  Returns:
    Collection with values modified following `fn`.
  """
  return {k: fn(v) for k, v in x.items()}


def prefix_keys(x: Mapping[str, _Value], prefix: str, delimiter: str = "/") -> Mapping[str, _Value]:
  """Apply a prefix to all dictionary keys.

  Args:
    x: Mapping to modify.
    prefix: Prefix to add to mapping's keys.
    delimiter: Delimiter to place between prefix and mapping's original keys.

  Returns:
    Collection with prefixed keys.
  """
  if not prefix:
    return x
  return key_apply(x, lambda k: f"{prefix}{delimiter}{k}")


def unprefix_keys(x: Mapping[str, _Value], prefix: str, delimiter: str = "/") -> Mapping[str, _Value]:
  """Remove a prefix from all dictionary keys.

  Args:
    x: Mapping to modify.
    prefix: Prefix to remove from mapping's keys.
    delimiter: Delimiter placed between prefix and mapping's original keys.

  Returns:
    Collection with prefix removed from keys.
  """
  if not prefix:
    return x

  def _unprefix(k: str) -> str:
    """Remove a prefix from a key."""
    # This is a slower unprefix method to validate inputs.
    tokens = k.split(delimiter)
    assert (len(tokens) >= 2) and (tokens[0] == prefix)
    return delimiter.join(tokens[1:])

  return key_apply(x, _unprefix)
