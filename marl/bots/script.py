"""Bot that follows a predefined script of actions."""
from typing import Optional, Sequence, Tuple

import numpy as np

from marl import _types, individuals, worlds


class Script(individuals.Bot):
    """Loops through a script of actions."""

    def __init__(self, script: Sequence[int]):
        self._script = script
        self._dtype = type(script[0])
        self._t = 0

    def step(self, timestep: worlds.TimeStep, state: Optional[_types.Tree] = None) -> Tuple[_types.Tree, _types.State]:
        """Samples a random action."""
        del timestep, state
        self._t = (self._t + 1) % len(self._script)
        return np.asarray(self._script[self._t - 1], dtype=self._dtype), None

    def episode_reset(self, timestep: worlds.TimeStep):
        self._t = 0
        return None
