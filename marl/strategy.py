"""Strategy objects defining containers of policies and a distribution over them."""
from typing import List, Optional, Sequence, Tuple

import numpy as np

from marl import _types, individuals, worlds


class Strategy(individuals.Bot):
    """A collection of player policies and a distribution over their episodic play."""

    def __init__(
        self, policies: Sequence[individuals.Individual], mixture: Sequence[float], seed: Optional[int] = None
    ):
        """Initializes a new strategy."""
        self._policies = policies
        self._mixture = mixture
        self._rng = np.random.RandomState(seed)
        self._policy = None

        if (len(self._mixture) > 0) and (np.sum(self._mixture) - 1.0 > 0.0001):
            raise ValueError("Mixture must be a valid probability distribution.")
        if np.any(np.asarray(self._mixture) < 0):
            raise ValueError("Mixture must be a valid probability distribution.")

    def step(
        self, timestep: worlds.TimeStep, state: Optional[_types.Tree] = None, **kwargs
    ) -> Tuple[_types.Tree, _types.State]:
        """Selects an action to take given the current timestep."""
        assert self._policy, "Episode-reset must be called before step."
        return self._policy.step(timestep=timestep, state=state, **kwargs)

    def episode_reset(self, timestep: worlds.TimeStep, **kwargs):
        """Reset the state of the agent at the start of an epsiode.."""
        policy_index = self._rng.choice(len(self._policies), p=self._mixture)
        self.set_policy(policy_index)
        return self._policy.episode_reset(timestep=timestep, **kwargs)

    def set_policy(self, policy_id: int):
        """Set the policy for the next episode."""
        self._policy = self._policies[policy_id]

    def add_policy(self, policy):
        """Adds a new policy with zero support."""
        self._policies.append(policy)
        self._mixture = np.append(self._mixture, 0.0)

    @property
    def mixture(self) -> List[float]:
        """Getter for `mixture`."""
        return self._mixture

    @mixture.setter
    def mixture(self, value: Sequence[float]):
        """Setter for `mixture`."""
        self._mixture = value

    def __len__(self):
        """Length as the number of policies."""
        return len(self._policies)
