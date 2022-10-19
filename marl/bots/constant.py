"""Constant action bots."""
from typing import Optional, Tuple

import numpy as np

from marl import _types, individuals, worlds


class ConstantIntAction(individuals.Bot):
    """Bot that plays a random discrete integer action."""

    def __init__(self, action: int, env_spec: worlds.EnvironmentSpec):
        self._action = action
        if env_spec.action.dtype not in [int, np.int32, np.int64]:
            raise ValueError("`ConstantIntAction` expects the environment to specify int actions.")

    def step(self, timestep: worlds.TimeStep, state: Optional[_types.Tree] = None) -> Tuple[_types.Tree, _types.State]:
        """Samples a random action."""
        del timestep, state
        return np.asarray(self._action), None

    def episode_reset(self, timestep: worlds.TimeStep):
        return None
