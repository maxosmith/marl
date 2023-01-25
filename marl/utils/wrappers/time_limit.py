"""Limits episode length for a game. """
from typing import Any

import numpy as np

from marl import _types, worlds


class TimeLimit(worlds.Game):
    """Limits episode length for a game."""

    def __init__(self, game: worlds.Game, num_steps: int, observe: bool = False):
        """Initializes an instance of the `TimeLimit` wrapper.

        Args:
            game: game to wrap with a time limit.
            num_steps: maximum number of steps in an episode.
            observe: add the current time to the observation.
        """
        self._game = game
        self._num_steps = num_steps
        self._t = None
        self._observe = observe

    def reset(self) -> worlds.PlayerIDToTimestep:
        """Returns the first `TimeStep` of a new episode."""
        self._t = 0
        timesteps = self._game.reset()
        timesteps = self._maybe_observe_time(timesteps)
        return timesteps

    def step(self, actions: _types.PlayerIDToAction) -> worlds.PlayerIDToTimestep:
        """Updates the environment according to the action."""
        timesteps = self._game.step(actions)
        self._t += 1
        if self._t >= self._num_steps:
            for agent_id, agent_step in timesteps.items():
                timesteps[agent_id] = worlds.TimeStep(
                    step_type=worlds.StepType.LAST,
                    reward=agent_step.reward,
                    observation=agent_step.observation,
                )
        return self._maybe_observe_time(timesteps)

    def reward_specs(self) -> worlds.PlayerIDToSpec:
        """Describes the reward returned by the environment."""
        return self._game.reward_specs()

    def observation_specs(self) -> worlds.PlayerIDToSpec:
        """Describes the observations provided by the environment."""
        if not self._observe:
            return self._game.observation_specs()

        obs_specs = self._game.observation_specs()
        modified_specs = {}
        for player_id, obs_spec in obs_specs.items():
            modified_specs[player_id] = {
                "observation": obs_spec,
                "time": worlds.ArraySpec(shape=(), dtype=np.int32, name="time"),
            }
        return modified_specs

    def action_specs(self) -> worlds.PlayerIDToSpec:
        """Describes the actions that should be provided to `step`."""
        return self._game.action_specs()

    def render(self, *args, **kwargs) -> Any:
        """Render the environment."""
        return self._game.render(*args, **kwargs)

    def _maybe_observe_time(self, timesteps: worlds.PlayerIDToTimestep) -> worlds.PlayerIDToTimestep:
        """Maybe add time as a field of the observation."""
        if not self._observe:
            return timesteps

        for agent_id, agent_step in timesteps.items():
            observation = {
                "observation": agent_step.observation,
                "time": np.array(self._t, dtype=np.int32),
            }
            timesteps[agent_id] = agent_step._replace(observation=observation)
        return timesteps
