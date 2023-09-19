"""Wrapper that changes a multiagent interfnce (game) to single-agent (env)."""

import dataclasses

from marl import specs, types, worlds


@dataclasses.dataclass
class GameToEnv(worlds.Environment):
  """Wrapper that changes a multiagent interfance into single-agent.

  The main usecase for this wrapper is for games that can flexibly use any number
  of players. Then this wrapper can provide an interface between single-agent
  algorithms and the game.
  """

  game: worlds.Game
  player_id: types.PlayerID

  def reset(self) -> worlds.TimeStep:
    """Starts a new sequence and returns the first `TimeStep` of this sequence."""
    return self.game.reset()[self.player_id]

  def step(self, action: types.Action) -> worlds.TimeStep:
    """Updates the environment according to the action and returns a `TimeStep`."""
    return self.game.step({self.player_id: action})[self.player_id]

  def reward_spec(self) -> specs.TreeSpec:
    """Describes the reward returned by the environment."""
    self.game.action_specs()[self.player_id]

  def observation_spec(self) -> specs.TreeSpec:
    """Defines the observations provided by the environment."""
    return self.game.observation_specs()[self.player_id]

  def action_spec(self) -> specs.TreeSpec:
    """Defines the actions that should be provided to `step`."""
    return self.game.action_specs()[self.player_id]
