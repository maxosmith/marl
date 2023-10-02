"""Bot that plays a constant action."""
from typing import Tuple

from marl import individuals, types, worlds


class ConstantActionBot(individuals.Bot):
  """Base class for types that are able to interact in a world."""

  def __init__(self, action: types.Action) -> None:
    super().__init__()
    self._action = action

  def step(
      self,
      state: types.State,
      timestep: worlds.TimeStep,
  ) -> Tuple[types.Action, types.State]:
    """Selects an action to take given the current timestep."""
    del timestep
    return state, self._action

  def episode_reset(self, timestep: worlds.TimeStep) -> types.State:
    """Resets the agent's episodic state."""
    del timestep
    return ()
