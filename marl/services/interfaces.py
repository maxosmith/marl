"""Interfaces that can be implemented by services."""
import abc
from typing import Generic, List, Optional, Sequence, TypeVar

from marl import types

_T = TypeVar("_T")


class VariableSource(abc.ABC):
  """Abstract source of variables.

  Objects which implement this interface provide a source of variables, returned
  as a collection of (nested) numpy arrays. Generally this will be used to
  provide variables to some learned policy/etc.
  """

  @abc.abstractmethod
  def get_variables(self, names: Optional[Sequence[str]] = None) -> List[types.Tree]:
    """Return the named variables as a collection of (nested) numpy arrays.

    Args:
        names: args where each name is a string identifying a predefined subset of
            the variables.

    Returns:
        A list of (nested) numpy arrays `variables` such that `variables[i]`
        corresponds to the collection named by `names[i]`.
    """


class Saveable(abc.ABC, Generic[_T]):
  """An interface for saveable objects."""

  @abc.abstractmethod
  def save(self) -> _T:
    """Returns the state from the object to be saved."""

  @abc.abstractmethod
  def restore(self, state: _T):
    """Given the state, restores the object."""


class Worker(abc.ABC):
  """An interface for (potentially) distributed workers."""

  @abc.abstractmethod
  def run(self, *args, **kwargs):
    """Runs the worker."""
