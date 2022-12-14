"""Limits episode length for a game. """
from typing import Any

from marl import _types, worlds


class TimeLimit(worlds.Game):
    """Limits episode length for a game."""

    fields = ("env", "max_steps_per_episode", "_t")

    def __init__(self, game: worlds.Game, num_steps: int):
        """Initializes an instance of the `TimeLimit` wrapper.

        Args:
            game: game to wrap with a time limit.
            num_steps: maximum number of steps in an episode.
        """
        self._game = game
        self._num_steps = num_steps
        self._t = None

    def reset(self) -> worlds.PlayerIDToTimestep:
        """Returns the first `TimeStep` of a new episode."""
        self._t = 0
        return self._game.reset()

    def step(self, actions: _types.PlayerIDToAction) -> worlds.PlayerIDToTimestep:
        """Updates the environment according to the action."""
        timestep = self._game.step(actions)
        self._t += 1
        if self._t >= self._num_steps:
            for agent_id, agent_step in timestep.items():
                timestep[agent_id] = worlds.TimeStep(
                    step_type=worlds.StepType.LAST,
                    reward=agent_step.reward,
                    observation=agent_step.observation,
                )
            return timestep
        else:
            return timestep

    def reward_specs(self) -> worlds.PlayerIDToSpec:
        """Describes the reward returned by the environment."""
        return self._game.reward_specs()

    def observation_specs(self) -> worlds.PlayerIDToSpec:
        """Describes the observations provided by the environment."""
        return self._game.observation_specs()

    def action_specs(self) -> worlds.PlayerIDToSpec:
        """Describes the actions that should be provided to `step`."""
        return self._game.action_specs()

    def render(self, *args, **kwargs) -> Any:
        """Render the environment."""
        return self._game.render(*args, **kwargs)
