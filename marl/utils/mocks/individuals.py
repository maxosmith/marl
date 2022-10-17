from typing import Tuple

from marl import _types, individuals, worlds
from marl.utils import spec_utils


class Bot(individuals.Bot):
    """Mock bot that plays randomly conforming to an environment spec."""

    def __init__(self, env_spec: worlds.EnvironmentSpec):
        self._env_spec = env_spec

    def step(self, timestep: worlds.TimeStep, state: _types.State) -> Tuple[_types.Action, _types.State]:
        """Samples a random action."""
        spec_utils.validate_spec(self._env_spec.observation, timestep.observation)
        return spec_utils.generate_from_spec(self._env_spec.action)


class Agent(individuals.Agent):
    """Mock agent that plays randomly conforming to an environment spec."""

    def __init__(self, env_spec: worlds.EnvironmentSpec):
        self._env_spec = env_spec

    def step(self, timestep: worlds.TimeStep, state: _types.State) -> Tuple[_types.Action, _types.State]:
        """Samples a random action."""
        spec_utils.validate_spec(self._env_spec.observation, timestep.observation)
        return spec_utils.generate_from_spec(self._env_spec.action)

    def update(self) -> None:
        pass
