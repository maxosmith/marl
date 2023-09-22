"""Test for directory utility functions."""
import pathlib
import shutil

from absl.testing import absltest, parameterized

from marl.utils import file_utils


class DirectoryUtilityTest(parameterized.TestCase):
  """Test suite for directory utility functions."""

  def setUp(self):
    """Setup a test."""
    super().setUp()
    # Create a temporary directory for testing
    self.temp_dir = pathlib.Path("temp_test_dir")
    self.temp_dir.mkdir(exist_ok=True)

    # Create some subdirectories inside the temporary directory
    (self.temp_dir / "subdir1").mkdir()
    (self.temp_dir / "subdir2").mkdir()

  def tearDown(self):
    """Teardown a test."""
    super().tearDown()
    # Remove the temporary directory after each test (if it still exists)
    if self.temp_dir.exists():
      shutil.rmtree(self.temp_dir)

  def test_rm_dir(self):
    """Tests rm_dir function."""
    self.assertTrue(self.temp_dir.exists())
    file_utils.rm_dir(self.temp_dir)
    self.assertFalse(self.temp_dir.exists())

  @parameterized.parameters(("temp_test_dir", ["subdir1", "subdir2"]), ("temp_test_dir/subdir1", []))
  def test_get_subdirs(self, path, expected_subdirs):
    """Tests get_subdirs function."""
    subdirs = file_utils.get_subdirs(path)
    self.assertListEqual(
        sorted([s.name for s in subdirs]),
        sorted(expected_subdirs),
    )


if __name__ == "__main__":
  absltest.main()
