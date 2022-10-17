from typing import Callable, Optional

import tensorflow as tf

from marl.utils.loggers import base


def pascal_case(key: str) -> str:
    """Format keys according to Pascal case (e.g., PascalCase)."""
    return key.title().replace("_", "")


class TensorboardLogger(base.Logger):
    """Logger that writes data as Tensorboard summaries (event files)."""

    def __init__(
        self, log_dir: str, steps_key: Optional[str] = None, key_format_fn: Optional[Callable[[str], str]] = None
    ):
        """Initializes a TensorboardLogger.

        Args:
            log_dir: directory in which to write log files.
            steps_key: key in the log's data referring to the current step.
                If not specified then each call to the logger is assumed to be a step.
            key_format_fn: preprocess data keys before writing to the log.
        """
        self._summary_writer = tf.summary.create_file_writer(log_dir)
        self._step_key = steps_key
        self._key_format_fn = key_format_fn
        self._num_calls = 0

    def write(self, data: base.LogData):
        step = data[self._step_key] if self._step_key is not None else self._num_calls

        with self._summary_writer.as_default():
            for key, value in data.items():
                key = self._key_format_fn(key) if self._key_format_fn else key
                tf.summary.scalar(key, data=value, step=step)

        self._num_calls += 1

    def close(self):
        self._summary_writer.close()
