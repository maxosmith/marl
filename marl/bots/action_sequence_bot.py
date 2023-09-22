"""Bot that follows a predefined sequence of actions."""
from typing import Sequence, Tuple

from marl import individuals, types, worlds


class ActionSequenceBot(individuals.Bot):
  """Bot that follows a predefined sequence of actions."""

  def __init__(self, sequence: Sequence[types.Action]) -> None:
    super().__init__()
    self._sequence = sequence
    self._sequence_len = len(self._sequence)
    self._t = None

  def step(
      self,
      state: types.State,
      timestep: worlds.TimeStep,
  ) -> Tuple[types.Action, types.State]:
    """Selects an action to take given the current timestep."""
    del timestep
    if self._t is None:
      raise RuntimeError("`episode_reset` must be called before `step`.")
    action = self._sequence[self._t % self._sequence_len]
    self._t += 1
    return state, action

  def episode_reset(self, timestep: worlds.TimeStep):
    """Resets the agent's episodic state."""
    del timestep
    self._t = 0
    return ()
