"""Test for `TODO`."""
import dataclasses
from typing import Sequence

from absl.testing import absltest, parameterized

from marl.utils.loggers import base_logger, composite_logger


@dataclasses.dataclass
class MockLogger(base_logger.BaseLogger):
  """Mock logger for testing."""

  expected_data: Sequence[base_logger.LogData]

  def __post_init__(self):
    """Post initializer."""
    self._write_i: int = 0

  def write(self, data: base_logger.LogData):
    """Write a batch of data."""
    expected = self.expected_data[self._write_i]
    self._write_i += 1

    for key, value in expected.items():
      assert key in data
      assert value == data[key]

  def close(self):
    """Close the logger's resources."""
    assert self._write_i == len(self.expected_data)


class CompositeLoggerTest(parameterized.TestCase):
  """Test suite for `CompositeLogger`."""

  @parameterized.parameters(
      ([{"key1": "value1", "key2": 2}],),
      ([{"keyA": [1, 2, 3], "keyB": "valueB"}, {"key1": "value1", "key2": 2}],),
  )
  def test_logger(self, data):
    """Tests the logger's write method."""
    logger = [MockLogger(data), MockLogger(data)]
    logger = composite_logger.CompositeLogger(loggers=logger)
    for datum in data:
      logger.write(datum)
    logger.close()


if __name__ == "__main__":
  absltest.main()
