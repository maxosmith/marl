"""Logger that does not record any data."""
from marl.services.arenas.loggers import base_logger


class NoopLogger(base_logger.BaseLogger):
  """Logger that does not record any data."""

  def write(self, data: base_logger.LogData):
    """Writes `data` to destination (file, terminal, database, etc.)."""
    del data

  def close(self):
    """Closes the logger and underlying services."""
    pass
