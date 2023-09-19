"""Bot that randomly selects actions."""
from typing import Tuple

import numpy as np

from marl import individuals, types, worlds


class RandomActionBot(individuals.Bot):
    """Bot that randomly selects actions."""

    def __init__(self, num_actions: int, seed: int | None = None):
        """Initializer."""
        self._num_actions = num_actions
        self._rng = np.random.default_rng(seed)

    def step(
        self,
        state: types.State,
        timestep: worlds.TimeStep,
    ) -> Tuple[types.Action, types.State]:
        """Selects an action to take given the current timestep."""
        del timestep
        return state, self._rng.choice(self._num_actions)

    def episode_reset(self, timestep: worlds.TimeStep):
        """Resets the agent's episodic state."""
        del timestep
        return ()
