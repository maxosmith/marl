from typing import Tuple

from marl import _types, individuals, worlds


class Bot(individuals.Bot):
    """Casts a policy as a `Bot`."""

    def __init__(self, policy):
        self._policy = policy

    def step(self, timestep: worlds.TimeStep, state: _types.State) -> Tuple[_types.Action, _types.State]:
        return self._policy.step(timestep, state)

    def episode_reset(self, timestep: worlds.TimeStep):
        return None
