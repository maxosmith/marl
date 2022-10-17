import collections
import time
from typing import Any, Callable, List, Mapping, Optional, Sequence, Union


class Stopwatch:
    """Stopwatch that maintains a collection of splits."""

    def __init__(self):
        # List of splits for each key.
        self._times: Mapping[str, List] = collections.defaultdict(list)
        # Whether the stop-watch is running/active for each key.
        self._active: Mapping[str, bool] = {}

    def start(self, key: str):
        if (key in self._active) and self._active[key]:
            raise ValueError(f"Stopwatch already running for `{key}'.")

        self._times[key].append(time.time())
        self._active[key] = True

    def stop(self, key: str):
        if (key not in self._active) or (not self._active[key]):
            raise ValueError(f"Stopwatch not started for `{key}'.")

        self._times[key][-1] = time.time() - self._times[key][-1]
        self._active[key] = False

    def split(self, key: str):
        self.stop(key)
        self.start(key)

    def get_splits(self, aggregate_fn: Optional[Callable[[Sequence[float]], Any]] = None) -> Mapping[str, Any]:
        if not aggregate_fn:
            return self._times
        return {key: aggregate_fn(value) for key, value in self._times.items()}

    def clear(self, key: Optional[Union[str, Sequence[str]]] = None):
        if not key:
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
