import dataclasses

import numpy as np

from marl import bots, individuals, types, worlds


@dataclasses.dataclass
class AgentState:
  """Agent state for `EpsilonRandomWrapper`.

  Logits are included in its state to allow for Fake testing of stochastic policies.
  """

  logits: np.ndarray


class EpsilonRandomWrapper(individuals.Bot):
  """Bot that randomly selects actions."""

  def __init__(
      self,
      bot: individuals.Bot,
      num_actions: int,
      epsilon: float,
      seed: int | None = None,
  ):
    """Initializer."""
    self._bot = bot
    self._bot_state = None
    self._epsilon = epsilon
    self._rng = np.random.default_rng(seed)
    self._rand_bot = bots.RandomActionBot(num_actions)

  def step(
      self,
      state: types.State,
      timestep: worlds.TimeStep,
  ) -> tuple[types.State, types.Action]:
    """Selects an action to take given the current timestep."""
    # Action selection.
    self._bot_state, bot_act = self._bot.step(self._bot_state, timestep)
    rand_state, rand_act = self._rand_bot.step(state, timestep)
    action = rand_act if self._rng.random() < self._epsilon else bot_act
    # Compute composite action logits.
    logits = rand_state.logits * self._epsilon
    logits[action] += 1 - self._epsilon
    return AgentState(logits=logits), action

  def episode_reset(self, timestep: worlds.TimeStep) -> types.State:
    """Resets the agent's episodic state."""
    self._bot_state = self._bot.episode_reset(timestep)
    return self._rand_bot.episode_reset(timestep)
