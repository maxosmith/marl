"""Observe the info state from an `OpenSpiel` game."""
from typing import Any

from marl import specs, types, worlds


class OpenSpielInfoState(worlds.Game):
  """Observe the info state from an `OpenSpiel` game.

  Args:
    game: `OpenSpiel` game.
  """

  def __init__(self, game: worlds.Game):
    """Initializes an instance of the `PreprocessOpenSpiel` wrapper."""
    self._game = game

  def reset(self) -> worlds.PlayerIDToTimestep:
    """Returns the first `TimeStep` of a new episode."""
    timesteps = self._game.reset()
    return self._preprocess(timesteps)

  def step(self, actions: types.PlayerIDToAction) -> worlds.PlayerIDToTimestep:
    """Updates the environment according to the action."""
    timesteps = self._game.step(actions)
    return self._preprocess(timesteps)

  def reward_specs(self) -> specs.PlayerIDToSpec:
    """Describes the reward returned by the environment."""
    return self._game.reward_specs()

  def observation_specs(self) -> specs.PlayerIDToSpec:
    """Describes the observations provided by the environment."""
    obs_specs = self._game.observation_specs()
    modified_specs = {}
    for player_id, obs_spec in obs_specs.items():
      modified_specs[player_id] = obs_spec["info_state"]
    return modified_specs

  def action_specs(self) -> specs.PlayerIDToSpec:
    """Describes the actions that should be provided to `step`."""
    return self._game.action_specs()

  def render(self, *args, **kwargs) -> Any:
    """Render the environment."""
    return self._game.render(*args, **kwargs)

  def _preprocess(self, timesteps: worlds.PlayerIDToTimestep) -> worlds.PlayerIDToTimestep:
    """Maybe add time as a field of the observation."""
    for agent_id, agent_step in timesteps.items():
      timesteps[agent_id] = agent_step._replace(observation=agent_step.observation["info_state"])
    return timesteps
