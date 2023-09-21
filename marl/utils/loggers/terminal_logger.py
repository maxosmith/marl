"""Log data to a string buffer (default is to stdout)."""
import logging
import time
from typing import Any, Callable

import numpy as np

from marl.utils import tree_utils
from marl.utils.loggers import base_logger

PrintFn = Callable[[str], None]
StringifyFn = Callable[[base_logger.LogData], str]


def _format_value(value: Any) -> str:
  """Internal function for formatting values."""
  value = tree_utils.to_numpy(value)
  if isinstance(value, (float, np.number)):
    return f"{value:0.3f}"
  return f"{value}"


def data_to_table(data: base_logger.LogData, precision: int = 4, padding: int = 3) -> str:
  """Converts `data` to a printed table of text."""
  render = "\n"  # Start table on a newline.
  key_size = max([len(k) for k in data.keys()])
  val_size = max([len(str(int(v))) for v in data.values()])
  val_size += 1 + precision + padding  # 1 for sign.

  for key, value in data.items():
    render += f"{key:<{key_size}} {value:>{val_size}.{precision}f}\n"

  return render


def data_to_string(data: base_logger.LogData, delimiter: str = "|") -> str:
  """Converts `data` to a printed key-value line of text."""
  return f" {delimiter} ".join(f"{k}: {_format_value(v)}" for k, v in data.items())


class TerminalLogger(base_logger.BaseLogger):
  """Log data to a string buffer.

  Args:
    print_fn: function that prints to the desired output stream.
    stringify_fn: function that transforms `LogData` into a format that
      a human can read from the terminal.
    frequency_secs: how often (in seconds) to wait between terminal prints.
  """

  def __init__(
      self,
      *,
      print_fn: PrintFn = logging.info,
      stringify_fn: StringifyFn = data_to_table,
      frequency_secs: float = 0.0,
  ):
    """Initializes a TerminalLogger."""
    self._print_fn = print_fn
    self._stringify_fn = stringify_fn
    self._frequency_secs = frequency_secs
    self._time_of_last_write = time.time()

  def write(self, data: base_logger.LogData):
    """Write a batch of data."""
    now = time.time()
    if (now - self._time_of_last_write) > self._frequency_secs:
      self._print_fn(self._stringify_fn(data))
      self._time_of_last_write = now

  def close(self):
    """Close the logger's resources."""
    pass
