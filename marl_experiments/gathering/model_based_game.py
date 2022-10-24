"""."""
import numpy as np

from marl import types, worlds
from marl.games import gathering


class ModelBasedGame(worlds.Game):
    """."""

    def __init__(self, true_game: worlds.Game):
        """Initializes."""
        self._true_game = true_game

    def reset(self) -> worlds.PlayerIDToTimestep:
        """Reset the game to it's initial state."""

    def step(self, actions: types.PlayerIDToAction) -> worlds.PlayerIDToTimestep:
        """Updates the game according to eahc players' action."""

    def reward_specs(self) -> worlds.PlayerIDToSpec:
        """Describes the reward returned by the environment."""
        return self._true_game.reward_specs()

    def observation_specs(self) -> worlds.PlayerIDToSpec:
        """Describes the observations provided by the environment."""
        return self._true_game.observation_specs()

    def action_specs(self) -> worlds.PlayerIDToSpec:
        """Describes the actions that should be provided to `step`."""
        return self._true_game.action_specs()
