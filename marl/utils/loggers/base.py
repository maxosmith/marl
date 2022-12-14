import abc
from typing import Any, Mapping

LogData = Mapping[str, Any]


class Logger(abc.ABC):

    def __call__(self, data: LogData):
        self.write(data)

    @abc.abstractmethod
    def write(self, data: LogData):
        """Writes `data` to destination (file, termina, database, etc.)."""

    @abc.abstractmethod
    def close(self):
        """Closes the logger and underlying services."""


class NoOpLogger(Logger):
    """Simple `Logger` that does nothing."""

    def write(self, data: LogData):
        del data

    def close(self):
        pass
