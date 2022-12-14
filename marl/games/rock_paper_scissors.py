"""RockPaperScissors game."""
import enum

import numpy as np

from marl import _types, worlds


class RockPaperScissorsActions(enum.IntEnum):
    """Actions available in `RockPaperScissors`."""

    ROCK = 0
    PAPER = 1
    SCISSORS = 2


class RockPaperScissors(worlds.Game):
    """Rock-Paper-Scissors."""

    _M = np.array(
        [[[0, 0], [-1, 1], [1, -1]], [[1, -1], [0, 0], [-1, 1]], [[-1, 1], [1, -1], [0, 0]]], dtype=np.float32
    )
    _NUM_PLAYERS = 2

    def __init__(self, num_stages: int = 1):
        """Initializes an instance of the `RockPaperScissors` game.

        Args:
            num_stages: Number of stages (rounds) of the game to play.
        """
        self._num_stages = num_stages
        self._stage_i = None

    def reset(self) -> worlds.PlayerIDToTimestep:
        """Reset the game to it's initial state."""
        self._stage_i = 0
        return {
            id: worlds.TimeStep(
                step_type=worlds.StepType.FIRST,
                reward=np.asarray(0.0, dtype=np.float32),
                observation=np.zeros([6], dtype=np.int32),
            )
            for id in range(RockPaperScissors._NUM_PLAYERS)
        }

    def step(self, actions: _types.PlayerIDToAction) -> worlds.PlayerIDToTimestep:
        """Updates the game according to eahc players' action."""
        if len(actions) != 2:
            raise ValueError("RPS only supports 2-players.")
        if (0 not in actions) or (1 not in actions):
            raise ValueError("RPS only suppers Player IDs: {0, 1}.")

        row_action = actions[0]
        col_action = actions[1]
        row_reward, col_reward = RockPaperScissors._M[row_action, col_action]
        next_observation = np.zeros([6], dtype=np.int32)
        next_observation[row_action] = 1
        next_observation[col_action + 3] = 1

        self._stage_i += 1
        step_type = worlds.StepType.LAST if self._stage_i >= self._num_stages else worlds.StepType.MID

        return {
            0: worlds.TimeStep(
                step_type=step_type,
                reward=row_reward,
                observation=next_observation,
            ),
            1: worlds.TimeStep(
                step_type=step_type,
                reward=col_reward,
                observation=next_observation,
            ),
        }

    def reward_specs(self) -> worlds.PlayerIDToSpec:
        """Describes the reward returned by the environment."""
        return {
            id: worlds.ArraySpec(shape=(), dtype=np.float32, name="reward")
            for id in range(RockPaperScissors._NUM_PLAYERS)
        }

    def observation_specs(self) -> worlds.PlayerIDToSpec:
        """Describes the observations provided by the environment."""
        return {
            id: worlds.ArraySpec(shape=(6,), dtype=np.int32, name="observation")
            for id in range(RockPaperScissors._NUM_PLAYERS)
        }

    def action_specs(self) -> worlds.PlayerIDToSpec:
        """Describes the actions that should be provided to `step`."""
        return {
            id: worlds.DiscreteArraySpec(dtype=np.int32, num_values=3, name="action")
            for id in range(RockPaperScissors._NUM_PLAYERS)
        }
