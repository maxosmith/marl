"""Utility functions for operating on files."""
import pathlib
import shutil
from typing import List


def rm_dir(path: str | pathlib.Path):
  """Remove directory recursively.

  Args:
    path: Directory to remove.
  """
  shutil.rmtree(path)


def get_subdirs(path: str | pathlib.Path) -> List[pathlib.Path]:
  """Get all of the immediate subdirectories in a folder."""
  return [child for child in pathlib.Path(path).iterdir() if child.is_dir()]
