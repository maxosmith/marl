"""Class that manages many `Logger` instances."""
import dataclasses
from typing import Sequence

from marl.utils.loggers import base_logger


@dataclasses.dataclass
class CompositeLogger(base_logger.BaseLogger):
  """Log to many loggers through a single composite logger interface.

  Args:
    loggers: collection of loggers to receive all data.
  """

  loggers: Sequence[base_logger.BaseLogger]

  def write(self, data: base_logger.LogData):
    """Write a batch of data."""
    for logger in self.loggers:
      logger.write(data)

  def close(self):
    """Close the logger's resources."""
    for logger in self.loggers:
      logger.close()
