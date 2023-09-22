"""Test for `marl.utils.import_utils`."""
import datetime
import pathlib

from absl.testing import absltest

from marl.utils import import_utils


class ImportUtilsTest(absltest.TestCase):
  """Test suite for `marl.utils.import_utils`."""

  def test_str_to_class(self):
    """Tests str_to_class function."""
    datetime_class = import_utils.str_to_class("datetime.datetime")
    self.assertEqual(datetime_class.__name__, "datetime")

  def test_str_to_class_with_pathlib(self):
    """Tests str_to_class function with pathlib.Path as input."""
    datetime_class = import_utils.str_to_class(pathlib.Path("datetime.datetime"))
    self.assertEqual(datetime_class.__name__, "datetime")

  def test_initialize(self):
    """Tests initialize function with kwargs."""
    date_instance = import_utils.initialize("datetime.date", year=2023, month=9, day=19)
    self.assertTrue(isinstance(date_instance, datetime.date))
    self.assertEqual(date_instance.year, 2023)
    self.assertEqual(date_instance.month, 9)
    self.assertEqual(date_instance.day, 19)


if __name__ == "__main__":
  absltest.main()
