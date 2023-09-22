"""Forces a game to be symmetric by sampling player IDs each episode."""
from typing import Any

import numpy as np

from marl import specs, types, worlds


class Symmetric(worlds.Game):
  """Converts a game into a symmetric game.

  The main usecase for this wrapper is for games that are already symmetric
  in nature, but their underlying implementation doesn't shuffle player IDs.

  Args:
    game: game to make symmetric.
  """

  def __init__(self, game: worlds.Game):
    """Initializes an instance of the `Symmetric` wrapper."""
    self._game = game
    self._id_map = {}

  def reset(self) -> worlds.PlayerIDToTimestep:
    """Returns the first `TimeStep` of a new episode."""
    timesteps = self._game.reset()
    original_keys = np.array(list(timesteps.keys()))
    new_keys = original_keys.copy()
    np.random.shuffle(new_keys)
    self._id_map = {k: v for k, v in zip(original_keys, new_keys)}
    return self._map(timesteps)

  def step(self, actions: types.PlayerIDToAction) -> worlds.PlayerIDToTimestep:
    """Updates the environment according to the action."""
    actions = self._map(actions, reverse=True)
    timesteps = self._game.step(actions)
    return self._map(timesteps)

  def reward_specs(self) -> specs.PlayerIDToSpec:
    """Describes the reward returned by the environment."""
    return self._game.reward_specs()

  def observation_specs(self) -> specs.PlayerIDToSpec:
    """Describes the observations provided by the environment."""
    return self._game.observation_specs()

  def action_specs(self) -> specs.PlayerIDToSpec:
    """Describes the actions that should be provided to `step`."""
    return self._game.action_specs()

  def render(self, *args, **kwargs) -> Any:
    """Render the environment."""
    return self._game.render(*args, **kwargs)

  def _map(self, data, reverse: bool = False):
    """Maybe add time as a field of the observation."""
    if reverse:
      return {k: data[v] for v, k in self._id_map.items()}
    else:
      return {k: data[v] for k, v in self._id_map.items()}
