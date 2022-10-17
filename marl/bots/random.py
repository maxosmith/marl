"""Random bots."""
from typing import Optional, Tuple

import numpy as np

from marl import _types, individuals, worlds


class RandomIntAction(individuals.Bot):
    """Bot that plays a random discrete integer action."""

    def __init__(self, num_actions: int, env_spec: worlds.EnvironmentSpec):
        self._num_actions = num_actions
        if env_spec.action.dtype not in [int, np.int32, np.int64]:
            raise ValueError("`RandomIntAction` expects the environment to specify int actions.")

    def step(self, timestep: worlds.TimeStep, state: Optional[_types.Tree] = None) -> Tuple[_types.Tree, _types.State]:
        """Samples a random action."""
        del timestep, state
        return np.random.choice(self._num_actions), None

    def episode_reset(self, timestep: worlds.TimeStep):
        return None
