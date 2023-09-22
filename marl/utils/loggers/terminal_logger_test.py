"""Test for TerminalLogger and utility functions."""
import functools
import time
from typing import Sequence

from absl.testing import absltest, parameterized

from marl.utils.loggers import base_logger, terminal_logger


class TerminalLoggerTest(parameterized.TestCase):
  """Test suite for TerminalLogger."""

  def setUp(self):
    """Set-up before a test case."""
    super().setUp()
    self.logged_data = None

    def mock_print_fn(data_: base_logger.LogData):
      """Mock print function for testing."""
      self.logged_data = data_

    self.mock_print = mock_print_fn

  @parameterized.parameters(
      dict(
          data=[{"a": 10.5, "b": 5.123456}],
          stringify_fn=terminal_logger.data_to_table,
          expected_outputs=["\na    10.5000\nb     5.1235\n"],
      ),
      dict(
          data=[{"a": 10.5, "b": 5.123456}],
          stringify_fn=terminal_logger.data_to_string,
          expected_outputs=["a: 10.5 | b: 5.123456"],
      ),
      dict(
          data=[{"a": 10.5, "b": 5.123456}, {"c": 44.02222, "d": 42}],
          stringify_fn=terminal_logger.data_to_table,
          expected_outputs=[
              "\na    10.5000\nb     5.1235\n",
              "\nc    44.0222\nd    42.0000\n",
          ],
      ),
  )
  def test_write(
      self,
      data: base_logger.LogData,
      stringify_fn: terminal_logger.StringifyFn,
      expected_outputs: Sequence[str],
  ):
    """Tests write."""
    logger = terminal_logger.TerminalLogger(
        print_fn=self.mock_print,
        stringify_fn=stringify_fn,
    )
    for datum, expected_output in zip(data, expected_outputs):
      time.sleep(0.001)  # Need to ensure that timestamps are different.
      logger.write(datum)
      self.assertEqual(self.logged_data, expected_output)

  @parameterized.parameters(
      dict(
          data=[{"a": 10.5, "b": 5.123456}],
          expected_outputs=["a: 10.5 | b: 5.123456"],
          frequency_secs=0.0,
          wait_secs=0.001,
      ),
      dict(
          data=[{"a": 10.5}, {"b": 5.123456}],
          expected_outputs=["a: 10.5", "b: 5.123456"],
          frequency_secs=0.0,
          wait_secs=0.001,
      ),
      dict(
          data=[{"a": 10.5}, {"b": 5.123456}],
          expected_outputs=[None, None],
          frequency_secs=0.01,
          wait_secs=0.001,
      ),
  )
  def test_frequency(
      self,
      data: Sequence[base_logger.LogData],
      expected_outputs: Sequence[str],
      frequency_secs: float,
      wait_secs: float,
  ):
    """Tests frequency in write."""
    logger = terminal_logger.TerminalLogger(
        print_fn=self.mock_print,
        stringify_fn=terminal_logger.data_to_string,
        frequency_secs=frequency_secs,
    )

    for datum, expected_output in zip(data, expected_outputs):
      time.sleep(wait_secs)
      logger.write(datum)
      self.assertEqual(self.logged_data, expected_output)


if __name__ == "__main__":
  absltest.main()
