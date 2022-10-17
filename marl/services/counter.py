"""Maintain global running counts.

The primary motivation for this service is a as a way to share step counts between
training and evaluation services. This enables results to be plotted along the same axis.
"""
import threading
import time
from typing import Dict, Mapping, Optional, Union

from marl.utils import dict_utils

_Number = Union[int, float]


class Counter:
    """A simple counter object that can periodically sync with a parent."""

    def __init__(
        self,
        parent: Optional["Counter"] = None,
        prefix: str = "",
        time_delta: float = 1.0,
        return_only_prefixed: bool = False,
    ):
        """Initialize the counter.

        Args:
            parent: a Counter object to cache locally (or None for no caching).
            prefix: string prefix to use for all local counts.
            time_delta: time difference in seconds between syncing with the parent
                counter.
            return_only_prefixed: if True, and if `prefix` isn't empty, return counts
                restricted to the given `prefix` on each call to `increment` and
                `get_counts`. The `prefix` is stripped from returned count names.
        """

        self._parent = parent
        self._prefix = prefix
        self._time_delta = time_delta

        # Hold local counts and we'll lock around that.
        # These are counts to be synced to the parent and the cache.
        self._counts = {}
        self._lock = threading.Lock()

        # We'll sync periodically (when the last sync was more than self._time_delta
        # seconds ago.)
        self._cache = {}
        self._last_sync_time = 0.0

        self._return_only_prefixed = return_only_prefixed

    def increment(self, **counts: _Number) -> Dict[str, _Number]:
        """Increment a set of counters.

        Args:
            **counts: keyword arguments specifying count increments.

        Returns:
            The [name, value] mapping of all counters stored, i.e. this will also
            include counts that were not updated by this call to increment.
        """
        with self._lock:
            for key, value in counts.items():
                self._counts.setdefault(key, 0)
                self._counts[key] += value
        return self.get_counts()

    def get_counts(self) -> Dict[str, _Number]:
        """Return all counts tracked by this counter."""
        now = time.time()
        # TODO(b/144421838): use futures instead of blocking.
        if self._parent and (now - self._last_sync_time) > self._time_delta:
            with self._lock:
                counts = dict_utils.prefix_keys(self._counts, self._prefix)
                # Reset the local counts, as they will be merged into the parent and the
                # cache.
                self._counts = {}
            self._cache = self._parent.increment(**counts)
            self._last_sync_time = now

        # Potentially prefix the keys in the counts dictionary.
        counts = dict_utils.prefix_keys(self._counts, self._prefix)

        # If there's no prefix make a copy of the dictionary so we don't modify the
        # internal self._counts.
        if not self._prefix:
            counts = dict(counts)

        # Combine local counts with any parent counts.
        for key, value in self._cache.items():
            counts[key] = counts.get(key, 0) + value

        if self._prefix and self._return_only_prefixed:
            counts = dict(
                [
                    (key[len(self._prefix) + 1 :], value)
                    for key, value in counts.items()
                    if key.startswith(f"{self._prefix}_")
                ]
            )
        return counts

    def save(self) -> Mapping[str, Mapping[str, _Number]]:
        return {"counts": self._counts, "cache": self._cache}

    def restore(self, state: Mapping[str, Mapping[str, _Number]]):
        # Force a sync, if necessary, on the next get_counts call.
        self._last_sync_time = 0.0
        self._counts = state["counts"]
        self._cache = state["cache"]

    def get_steps_key(self) -> str:
        """Returns the key to use for steps by this counter."""
        if not self._prefix or self._return_only_prefixed:
            return "steps"
        return f"{self._prefix}_steps"
