"""Class that manages many `Logger` instances."""
import time
import warnings
from typing import Optional, Sequence

from marl.utils.loggers import base


class LoggerManager(base.Logger):
    """Manager for a collection of `Logger`s."""

    def __init__(
        self,
        loggers: Sequence[base.Logger],
        time_frequency: Optional[float] = None,
        step_frequency: Optional[float] = None,
        step_key: Optional[str] = None,
    ):
        """Initializes an instance of `LoggerManager`.

        Args:
            loggers: collection of `Logger`s.
            time_frequency: how often in seconds to wait between writes.
            step_frequency: how often in steps, see `step_key`, to wait between writes.
                If specified, `step_key` must also be specified.
            step_key: the key for the step value in the written data.
                If specified, `step_frequency` must also be specified.
        """
        if (time_frequency is not None) and (step_frequency is not None):
            raise warnings.warn("It is not recommended to filter logs based off both time and step frequency.")
        if (step_key is not None) and (step_frequency is None):
            raise ValueError("`step_frequency` must be specified if `step_key` is specified.")
        if (step_key is None) and (step_frequency is not None):
            raise ValueError("`step_key` must be specififed if `step_frequency` is specified.")

        self._loggers = loggers
        # Filter based off walltime.
        self._time_frequency = time_frequency
        self._time_of_last_write = time.time()
        # Filter based off timestep.
        self._step_frequency = step_frequency
        self._step_key = step_key
        self._step_of_last_write = None

    def write(self, data: base.LogData):
        if self._should_write(data):
            self._write(data)
            self._time_of_last_write = time.time()
            self._step_of_last_write = data[self._step_key] if self._step_key else None

    def close(self):
        for logger in self._loggers:
            logger.close()

    def _should_write(self, data: base.LogData) -> bool:
        should_write = True

        # Filter based off of elapsed walltime.
        if self._time_frequency:
            now = time.time()
            enough_time_elapsed = (now - self._time_of_last_write) > self._time_frequency
            should_write = should_write and enough_time_elapsed

        # Filter based off of step counts.
        if self._step_frequency:
            step = data[self._step_key]
            enough_step_elapsed = (step - self._step_of_last_write) > self._step_frequency
            should_write = should_write and enough_step_elapsed

        return should_write

    def _write(self, data: base.LogData):
        for logger in self._loggers:
            logger.write(data)
