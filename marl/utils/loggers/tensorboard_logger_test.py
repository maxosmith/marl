"""Test suite for `tensorboard_logger`."""
import os
import tempfile

import numpy as np
import tensorflow as tf
from absl.testing import absltest, parameterized
from tensorflow.python.summary import summary_iterator

from marl.utils.loggers import tensorboard_logger


class TensorboardLoggerTest(parameterized.TestCase):
  """Test suite for TensorboardLogger."""

  def setUp(self):
    """Set-up before a test case."""
    super().setUp()
    self.log_dir = tempfile.TemporaryDirectory()
    self.logger = tensorboard_logger.TensorboardLogger(log_dir=self.log_dir.name)

  def tearDown(self):
    """Tear down resources after a test case."""
    super().tearDown()
    self.logger.close()

  @parameterized.parameters(
      ({"key_1": 1.0}, "key_1", tf.summary.scalar),
      ({"key_2": [1.0, 2.0]}, "key_2", tf.summary.histogram),
      ({"key_3": np.random.rand(10, 10)}, "key_3", tf.summary.image),
      ({"key_4": np.random.rand(10, 10, 3)}, "key_4", tf.summary.image),
  )
  def test_write(self, data, key, expected_tf_summary_func):
    """Tests write based on data shape."""
    self.logger.write(data)

    event_file = tf.io.gfile.glob(os.path.join(self.log_dir.name, "*"))[0]
    events = list(summary_iterator.summary_iterator(event_file))

    # Skip the first event which is about file_version.
    summary = events[1].summary.value[0]
    self.assertEqual(summary.tag, key)
    self.assertEqual(
        summary.metadata.plugin_data.plugin_name[:-1],  # Plugin name is plural.
        expected_tf_summary_func.__name__,
    )

  def test_close(self):
    """Tests closing of resources."""
    self.logger.close()

    # After closing, writing should raise an exception.
    with self.assertRaises(Exception):
      self.logger.write({"key": 1.0})


if __name__ == "__main__":
  absltest.main()
