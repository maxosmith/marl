"""Utility functions for operating on files."""
import os.path as osp
import shutil
import warnings


def get_extension(path: str) -> str:
    """Get the the extension for a file from its path."""
    return osp.splitext(path)[-1].lower()


def has_extension(path: str) -> bool:
    """Check if a file path has an extension."""
    return bool(osp.splitext(path)[-1])


def maybe_change_extension(path: str, new_extension: str) -> str:
    """Maybe change the extension of a filepath."""

    if has_extension(path):
        extension = get_extension(path)
        if extension == new_extension:
            return path
        else:
            warnings.warn(f"Changing the extension to {new_extension} of the path: {path}.")
            return f"{osp.splitext(path)[0]}.{new_extension}"
    else:
        return f"{osp.splitext(path)[0]}.{new_extension}"


def rm_dir(path: str) -> None:
    """Remove directory recursively."""
    shutil.rmtree(path)
