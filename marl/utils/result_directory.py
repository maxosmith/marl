"""Manages a directory to save results."""
import os
import os.path as osp

from marl.utils import file_utils


class ResultDirectory:
    """Establishes and manages a directory for writing results."""

    def __init__(self, dir: str, exist_ok: bool = False, overwrite: bool = False):
        """Initializes an instance of `ResultDirectory`.

        Args:
            dir: Directory path, if it does not already exist it will be created.
            exist_ok: If it is OK if the directory already exists.
            overwrite: If the directory already exists delete the existing directory.

        Raises:
            FileExistError if `exist_ok` is set to False and the directory already exists.
        """
        self.dir = dir
        if osp.exists(self.dir) and overwrite:
            file_utils.rm_dir(self.dir)
        os.makedirs(self.dir, exist_ok=exist_ok)

    def make_subdir(self, name: str) -> "ResultDirectory":
        """Create a `ResultDirectory` for a subdirectory."""
        return ResultDirectory(osp.join(self.dir, str(name)), exist_ok=False, overwrite=True)

    def subdir(self, name: str) -> "ResultDirectory":
        """Get a subdirectory."""
        return ResultDirectory(osp.join(self.dir, str(name)), exist_ok=True, overwrite=False)

    def file(self, name: str) -> str:
        """Get a file."""
        return osp.join(self.dir, name)

    def __str__(self) -> str:
        """String representation."""
        return dir
