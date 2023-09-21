"""Base logger interface."""
import abc
from typing import Any, Mapping

LogData = Mapping[str, Any]


class BaseLogger(abc.ABC):
  """Logger interface."""

  def __call__(self, data: LogData):
    """Writes `data` to destination (file, terminal, database, etc.)."""
    self.write(data)

  @abc.abstractmethod
  def write(self, data: LogData):
    """Writes `data` to destination (file, terminal, database, etc.)."""

  @abc.abstractmethod
  def close(self):
    """Closes the logger and underlying services."""
