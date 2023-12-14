"""Bot that randomly selects actions."""
import dataclasses
from typing import Tuple

import numpy as np

from marl import individuals, types, worlds


@dataclasses.dataclass
class AgentState:
  """Agent state for `RandomActionBot`.

  Logits are included in its state to allow for Fake testing of stochastic policies.
  """

  logits: np.ndarray


class RandomActionBot(individuals.Bot):
  """Bot that randomly selects actions."""

  def __init__(self, num_actions: int, seed: int | None = None):
    """Initializer."""
    self._num_actions = num_actions
    self._rng = np.random.default_rng(seed)
    self._state = AgentState(logits=np.ones((self._num_actions), dtype=float) / num_actions)

  def step(
      self,
      state: types.State,
      timestep: worlds.TimeStep,
  ) -> Tuple[types.Action, types.State]:
    """Selects an action to take given the current timestep."""
    del state
    if isinstance(timestep.observation, dict) and ("legal_actions" in timestep.observation):
      action_probs = timestep.observation["legal_actions"].astype(float)
      action_probs /= np.sum(action_probs)
      possible_actions = np.arange(timestep.observation["legal_actions"].shape[0])
      return AgentState(logits=action_probs), self._rng.choice(possible_actions, p=action_probs)
    else:
      return self._state, self._rng.choice(self._num_actions)

  def episode_reset(self, timestep: worlds.TimeStep) -> types.State:
    """Resets the agent's episodic state."""
    del timestep
    return self._state
