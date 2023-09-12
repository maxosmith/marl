"""Maintain global running counts.

The primary motivation for this service is a way to synchronize services. For example,
as a way to share step counts between training and evaluation services. This allows
results to be plotted along the same axis.
"""
import collections
import dataclasses
import threading
import time
from typing import Any, Mapping, Optional, Union

import numpy as np

from marl.utils import dict_utils

_Number = Union[int, float, np.number]
_Counts = Mapping[str, _Number]


@dataclasses.dataclass
class CounterState:
  """State of the counter.

  Args:
    counts: Increments that have happened since `cache` was updated by
      syncing with a potential parent `Counter`.
    cache: Total counts.
  """

  counts: _Counts
  cache: _Counts


class Counter:
  """Counting service.

  This counter can be either deployed as an independent service, or as a local
  counter that periodically syncs its counts with a `parent` remote `Counter`.

  Args:
    parent:
    prefix:
    time_delta_secs
  """

  def __init__(
      self,
      *,
      parent: Optional["Counter"] = None,
      prefix: str = "",
      time_delta_secs: float = 1.0,
  ):
    """Initialize a `Counter`."""
    self._parent = parent
    self._prefix = prefix
    self._time_delta_secs = time_delta_secs

    # Local counts that we will lock around access. These counts will be synced
    # to the parent and the cache. They track the counts since the last sync
    # with the parent.
    self._counts = collections.defaultdict(int)
    self._lock = threading.Lock()

    # If there is a parent we'll sync periodically counts. This can also be thought
    # of as the `total` counts from a cross-service perspective.
    self._cache = {}
    self._last_sync_time = 0.0

  def increment(self, **counts: _Number) -> _Counts:
    """Increment a set of counters.

    Args:
      counts: Keyword arguments specifying count increments.

    Returns:
      The [name, value] mapping of all counters stored.
    """
    with self._lock:
      for key, value in counts.items():
        self._counts[key] += value
    return self.get_counts()

  def get_counts(self) -> _Counts:
    """Return all counts tracked by this counter."""
    now_secs = time.time()
    if self._parent and (now_secs - self._last_sync_time) > self._time_delta_secs:
      with self._lock:
        counts = dict_utils.prefix_keys(self._counts, self._prefix)
        # Reset the local counts, as they will be merged into the parent and the cache.
        self._counts = {}
      self._cache = self._parent.increment(**counts)
      self._last_sync_time = now_secs

    # Potentially prefix the keys in the counts dictionary. If there's no prefix,
    # make a copy so we don't modify the internal `self._counts`.
    counts = dict_utils.prefix_keys(self._counts, self._prefix)
    if not self._prefix:
      counts = dict(counts)

    # Combine local counts with any parent counts.
    for key, value in self._cache.items():
      counts[key] = counts.get(key, 0) + value
    return counts

  def save(self) -> CounterState:
    """Return the state of the counter for saving."""
    return CounterState(counts=self._counts, cache=self._cache)

  def restore(self, state: CounterState):
    # Force a sync, if necessary, on the next `get_counts` call.
    self._last_sync_time = 0.0
    self._counts = state.counts
    self._cache = state.cache

  def reset(self):
    """Reset the state of the counter."""
    with self._lock:
      self._counts = {}
      self._cache = {}
      self._last_sync_time = 0.0
