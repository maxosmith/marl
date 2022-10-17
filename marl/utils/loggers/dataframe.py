import warnings

import pandas as pd

from marl.utils import file_utils
from marl.utils.loggers import base


class CSVLogger(base.Logger):
    """Logger that writes data to a CSV file on disk."""

    def __init__(self, path: str):
        self._path = file_utils.maybe_change_extension(path, "csv")

    def write(self, data: base.LogData):
        # Create the log file if it does not exist, otherwise append to it.
        with open(self._path, "a") as file:
            # Add the header only when the file is being created.
            pd.DataFrame(data).to_csv(file, header=file.tell() == 0)

    def close(self):
        pass

    @property
    def df(self) -> pd.DataFrame:
        return pd.read_csv(self._path)


class DataFrameInMemoryLogger(base.Logger):
    """Logger that maintains a DataFrame in memory and writes it to disk at termination.

    This logger is faster because it only writes to disk once; however, it does not checkpoint
    data throughout training so data may be lost.
    """

    def __init__(self, path: str):
        self._path = file_utils.maybe_change_extension(path, "csv")
        self._data = []

    def write(self, data: base.LogData):
        self._data.append(data)

    def close(self):
        with open(self._path, "w") as file:
            pd.DataFrame(self._data).to_csv(file)

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self._data)
