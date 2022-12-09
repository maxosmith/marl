import abc
from typing import Tuple

from marl import _types, worlds


class Individual(abc.ABC):
    """Base class for types that are able to interact in a world."""

    @abc.abstractmethod
    def step(self, timestep: worlds.TimeStep, state: _types.State) -> Tuple[_types.Action, _types.State]:
        """Selects an action to take given the current timestep."""

    @abc.abstractmethod
    def episode_reset(self, timestep: worlds.TimeStep):
        """Resets the agent's episodic state."""


class Bot(Individual):
    """Non-learning individual."""

    def update(self) -> None:
        """Maybe update the agent's internal parameters."""
        pass


class Agent(Individual):
    """Agents are individuals that maintain and update internal parameters."""

    @abc.abstractmethod
    def update(self) -> None:
        """Maybe update the agent's internal parameters."""
