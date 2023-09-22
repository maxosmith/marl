"""Test suite for the launchpad_utils module."""
import pathlib

from absl.testing import absltest, parameterized

from marl.utils import launchpad_utils


class TestLaunchpadUtils(parameterized.TestCase):
  """Test suite for the launchpad_utils module."""

  def setUp(self):
    """Setup function to create a testing environment before each test method is run.

    This includes:
        - Creating a temporary test directory
        - Setting up the paths for log file and output directory
    """
    self.test_dir = pathlib.Path("tests")
    self.test_dir.mkdir(exist_ok=True)
    self.log_file_path = self.test_dir / "test_log.log"
    self.output_dir = self.test_dir / "output"
    self.output_dir.mkdir(exist_ok=True)

  def tearDown(self):
    """Tear down function to clean up the testing environment after each test method is run.

    This includes:
        - Deleting all files and directories created in the test directory
        - Removing the test directory itself
    """
    for item in self.test_dir.iterdir():
      if item.is_file():
        item.unlink()
      else:
        for sub_item in item.iterdir():
          sub_item.unlink()
        item.rmdir()
    self.test_dir.rmdir()

  def test_split_log(self):
    """Test method to verify the functionality of the split_log function in the launchpad_utils module.

    This includes:
        - Creating a log file with some test logs
        - Running the split_log function with this log file
        - Asserting that the split_log function splits the logs correctly into separate files based on group names
    """
    with open(self.log_file_path, "w") as f:
      f.write("\x1b[1;32m[group_one/1] This is a test message from group one\x1b[0;0m\n")
      f.write("\x1b[1;32m[group_two/1] This is another test message from group two\x1b[0;0m\n")
      f.write("This is a message from main\n")

    launchpad_utils.split_log(self.log_file_path, self.output_dir)

    with open(self.output_dir / "group_one_1.log", "r") as f:
      self.assertEqual(f.read().strip(), "[group_one/1] This is a test message from group one")

    with open(self.output_dir / "group_two_1.log", "r") as f:
      self.assertEqual(f.read().strip(), "[group_two/1] This is another test message from group two")

    with open(self.output_dir / "main.log", "r") as f:
      self.assertEqual(f.read().strip(), "This is a message from main")


if __name__ == "__main__":
  absltest.main()
