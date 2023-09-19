"""Utilities for dependency injection."""
import importlib
from pathlib import Path
from typing import Any, Union


def str_to_class(path: Union[str, Path]) -> Any:
  """Convert a string or pathlib.Path representation of a class to the actual class object.

  This function accepts a module path to a class (as a string or pathlib.Path) and
  returns the class object.

  Args:
      path (Union[str, Path]): Module path to the class (e.g., 'module.ClassName' or pathlib.Path('module.ClassName')).

  Returns:
      Any: The class object referred to by the path.

  Raises:
      AttributeError: If the class cannot be found in the module.
      ModuleNotFoundError: If the module cannot be found.
  """
  tokens = str(path).split(".")
  module = ".".join(tokens[:-1])
  name = tokens[-1]
  module = importlib.import_module(module)
  return getattr(module, name)


def initialize(ctor_path: Union[str, Path], **kwargs: Any) -> Any:
  """Initialize an instance of a class using its module path.

  This function accepts a module path to a class (as a string or pathlib.Path) and keyword arguments,
  and returns an instance of the class, initialized with the given keyword arguments.

  Args:
      ctor_path (Union[str, Path]): Module path to the class constructor.
      **kwargs (Any): Keyword arguments to pass to the class constructor.

  Returns:
      Any: An instance of the class.
  """
  return str_to_class(ctor_path)(**kwargs)
