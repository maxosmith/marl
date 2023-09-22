"""Stopwatch."""
import collections
import time
from typing import Any, Callable, List, Mapping, Optional, Sequence, Union


class Stopwatch:
  """Stopwatch that maintains a collection of time splits.

  Args:
    buffer_size: number of splits to maintain per key.
  """

  def __init__(self, buffer_size: Optional[int] = None):
    """Initializer."""
    # List of splits for each key.
    self._times: Mapping[str, List] = collections.defaultdict(list)
    # Whether the stop-watch is running/active for each key.
    self._active: Mapping[str, bool] = {}
    self._buffer_size = buffer_size

  def start(self, key: str):
    """Start a new split.

    Args:
      key: name of the split.
    """
    if (key in self._active) and self._active[key]:
      raise ValueError(f"Stopwatch already running for `{key}'.")
    if self._buffer_size and len(self._times[key]) > self._buffer_size:
      self._times[key] = self._times[key][-self._buffer_size :]

    self._times[key].append(time.time())
    self._active[key] = True

  def stop(self, key: str):
    """Stop a running split.

    Args:
      key: name of the split.
    """
    if (key not in self._active) or (not self._active[key]):
      raise ValueError(f"Stopwatch not started for `{key}'.")

    self._times[key][-1] = time.time() - self._times[key][-1]
    self._active[key] = False

  def split(self, key: str):
    """Stop a split and immediately start a new one.

    Args:
      key: name of the split.
    """
    self.stop(key)
    self.start(key)

  def get_splits(
      self, aggregate_fn: Optional[Callable[[Sequence[float]], Any]] = None
  ) -> Mapping[str, Any]:
    """Get all of the splits recorded by this stopwatch.

    Args:
      aggregate_fn: optionally aggregate all of the splits for each key.

    Return:
      All splits, optionally aggregated.
    """

    def _noop(x):
      """Don't aggregate."""
      return x

    if aggregate_fn is None:
      aggregate_fn = _noop

    splits = {}
    for key, value in self._times.items():
      if self._active[key]:
        splits[key] = aggregate_fn(value[:-1])
      else:
        splits[key] = aggregate_fn(value)
    return splits

  def clear(self, key: Optional[Union[str, Sequence[str]]] = None):
    """Clear the stopwatch.

    Args:
      key: clear only the specified key instead of all keys.
    """
    if key is None:
      self._times = collections.defaultdict(list)
      self._active = {}
    elif isinstance(key, str):
      del self._times[key]
      del self._active[key]
    elif isinstance(key, Sequence):
      for unit_key in key:
        self.clear(unit_key)
    else:
      raise ValueError(f"Unsupported key type: {type(key)}.")
