"""Converts ravelled categorical observations into a grid."""
from typing import Any, Tuple

import numpy as np

from marl import _types, worlds


class ObservationVectorToGrid(worlds.Game):
    """Converts ravelled observations into a grid."""

    def __init__(self, game: worlds.Game, shape: Tuple[int, int], num_classes: int):
        """Initializes an instance of `ObservationVectorToGrid`.

        Args:
            game: The game to be wrapped.
            shape: Shape of the observation as a grid.
            num_classes: Number of possible classes each cell can take.
        """
        self._game = game
        self._shape = shape
        self._num_classes = num_classes

    def reset(self) -> worlds.PlayerIDToTimestep:
        """Returns the first `TimeStep` of a new episode."""
        timesteps = self._game.reset()
        return self._preprocess(timesteps)

    def step(self, actions: _types.PlayerIDToAction) -> worlds.PlayerIDToTimestep:
        """Updates the environment according to the action."""
        timesteps = self._game.step(actions)
        return self._preprocess(timesteps)

    def reward_specs(self) -> worlds.PlayerIDToSpec:
        """Describes the reward returned by the environment."""
        return self._game.reward_specs()

    def observation_specs(self) -> worlds.PlayerIDToSpec:
        """Describes the observations provided by the environment."""
        return {
            id: worlds.ArraySpec(shape=self._shape + (self._num_classes,), dtype=int, name="observation")
            for id in self._game.observation_specs().keys()
        }

    def action_specs(self) -> worlds.PlayerIDToSpec:
        """Describes the actions that should be provided to `step`."""
        return self._game.action_specs()

    def render(self, *args, **kwargs) -> Any:
        """Render the environment."""
        return self._game.render(*args, **kwargs)

    def _preprocess(self, timesteps: worlds.PlayerIDToTimestep) -> worlds.PlayerIDToTimestep:
        """Convert vector to categorical grid."""
        for agent_id, agent_step in timesteps.items():
            observation = agent_step.observation.astype(int)
            observation = np.reshape(observation, (self._num_classes,) + self._shape)
            observation = np.transpose(observation, (1, 2, 0))  # Class to trailling dimension.
            timesteps[agent_id] = agent_step._replace(observation=observation)
        return timesteps
