"""Log data to tensorboard."""
from typing import Callable, Optional

import numpy as np
import tensorflow as tf

from marl.utils.loggers import base_logger


class TensorboardLogger(base_logger.BaseLogger):
  """Logger that writes data as Tensorboard summaries (event files)."""

  def __init__(
      self,
      log_dir: str,
      step_key: Optional[str] = None,
      key_format_fn: Optional[Callable[[str], str]] = None,
  ):
    """Initializes a TensorboardLogger.

    Args:
        log_dir: directory in which to write log files.
        step_key: key in the log's data referring to the current step.
            If not specified then each call to the logger is assumed to be a step.
        key_format_fn: preprocess data keys before writing to the log.
    """
    self._summary_writer = tf.summary.create_file_writer(log_dir)
    self._step_key = step_key
    self._key_format_fn = key_format_fn
    self._num_calls = 0

  def write(self, data: base_logger.LogData):
    """Write a batch of data."""
    step = data.get(self._step_key, self._num_calls) if self._step_key is not None else self._num_calls
    with self._summary_writer.as_default():
      for key, value in data.items():
        num_dims = len(np.asarray(value).shape)
        key = self._key_format_fn(key) if self._key_format_fn else key
        if num_dims == 0:
          tf.summary.scalar(key, data=value, step=step)
        elif num_dims == 1:
          tf.summary.histogram(key, data=value, step=step)
        elif num_dims == 2:
          tf.summary.image(key, data=value[None, ..., None], step=step)
        elif num_dims == 3:
          tf.summary.image(key, data=value[None, ...], step=step)
        elif num_dims == 4:
          tf.summary.image(key, data=value, step=step)
        else:
          raise ValueError(f"Logging data with {num_dims} dims undefined.")

    self._num_calls += 1

  def close(self):
    """Close the logger's resources."""
    self._summary_writer.close()
