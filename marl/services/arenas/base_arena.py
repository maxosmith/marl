"""Interface for servies that simulate episodes."""
import abc
from typing import Optional


class BaseArena(abc.ABC):
  """Interface for an agent-environment interaction (arena) services."""

  @abc.abstractmethod
  def run(
      self,
      *,
      num_episodes: Optional[int] = None,
      num_timesteps: Optional[int] = None,
      **kwargs
  ):
    """Run the arena to generate experience.

    Runs either `num_episodes`, or _at least_ `num_timesteps`, or indefinitely.

    Args:
      num_episodes: Number of episodes to run.
      num_timesteps: Minimum number of timesteps to run.
    """

  @abc.abstractmethod
  def stop(self):
    """Stop running this service if set to run indefinitely."""
