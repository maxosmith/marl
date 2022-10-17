import abc
from typing import Generic, TypeVar

T = TypeVar("T")


class SaveableInterface(abc.ABC, Generic[T]):
    """An interface for saveable objects."""

    @abc.abstractmethod
    def save(self) -> T:
        """Returns the state from the object to be saved."""

    @abc.abstractmethod
    def restore(self, state: T):
        """Given the state, restores the object."""
