"""Dictionary utility operations."""
from typing import Any, Callable, Mapping


def key_apply(x: Mapping[Any, Any], fn: Callable[[Any], Any]) -> Mapping[Any, Any]:
  """Apply function over keys."""
  return {fn(k): v for k, v in x.items()}


def value_apply(x: Mapping[Any, Any], fn: Callable[[Any], Any]) -> Mapping[Any, Any]:
  """Apply function over values."""
  return {k: fn(v) for k, v in x.items()}


def prefix_keys(
    x: Mapping[str, Any], prefix: str, delimiter: str = "/"
) -> Mapping[str, Any]:
  """Apply a prefix to all dictionary keys."""
  if not prefix:
    return x
  return key_apply(x, lambda k: f"{prefix}{delimiter}{k}")


def unprefix_keys(
    x: Mapping[str, Any], prefix: str, delimiter: str = "/"
) -> Mapping[str, Any]:
  """Remove a prefix from all dictionary keys."""
  if not prefix:
    return x

  def _unprefix(k: str) -> str:
    """Remove a prefix from a key."""
    # This is a slower unprefix method to validate inputs.
    tokens = k.split(delimiter)
    assert (len(tokens) >= 2) and (tokens[0] == prefix)
    return delimiter.join(tokens[1:])

  return key_apply(x, _unprefix)
