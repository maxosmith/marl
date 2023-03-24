"""Empirical normal-form game. """
import dataclasses
import itertools
import shelve
from typing import Dict, Mapping, Optional, Union

import numpy as np

from marl import _types

_PureProfile = Dict[_types.PlayerID, _types.PolicyID]
_MixedProfile = Dict[_types.PlayerID, _types.Array]
_Profile = Union[_PureProfile, _MixedProfile]
_ProfilePayoffs = Dict[_types.PlayerID, _types.Array]
_ProfilePayoffSummary = Dict[_types.PlayerID, float]


@dataclasses.dataclass
class EmpiricalNFG:
    """A empirical normal-form game.

    Args:
        num_agents: The number of agents.
        file_backed_path: Optional path to a file for caching the NFG.
        flag: Flag for loading a shelve from disk.
    """

    num_agents: int
    path: Optional[str] = None
    flag: str = "c"

    def __post_init__(self):
        """Initialize a new normal form game."""
        # Payoffs associated with each strategy profile.
        if self.path:
            self.payoffs = shelve.open(self.path, flag=self.flag)
            self.num_policies = self._load_num_policies()
        else:
            self.payoffs = {}
            self.num_policies = {id: 0 for id in range(self.num_agents)}

    def __del__(self):
        """Destructor."""
        if self.path:
            self.payoffs.close()

    def add_payoffs(self, profile: _PureProfile, payoffs: _ProfilePayoffs):
        """Add payoff for a strategy profile.

        Args:
            profile: Map from AgentID to their PolicyID.
            payoff: List of payoffs for each agent.
        """
        key = self._profile_to_key(profile)

        # Log any new policies.
        for agent, policy in profile.items():
            if agent not in self.num_policies:
                self.num_policies[agent] = policy + 1  # Zero-indexed.
            else:
                self.num_policies[agent] = max(self.num_policies[agent], policy + 1)  # Zero-indexed.

        # Extend the list of samples.
        if key in self.payoffs:
            for agent in self.payoffs[key].keys():
                self.payoffs[key][agent] = np.append(self.payoffs[key][agent], payoffs[agent])

        # Start keeping track of a new profile.
        else:
            self.payoffs[key] = payoffs

    def average_payoffs(self, profile: _Profile) -> _ProfilePayoffSummary:
        """Get the expected payoffs for a strategy profile.

        Args:
            profile: Map from AgentID to either PolicyID or a mixture over policies.

        Returns:
            Dictionary from AgentID to the average payoff for that agent.
        """
        # Pure-strategy profile.
        if np.all([isinstance(p, _types.PolicyID) for p in profile.values()]):
            payoffs = self.payoff_samples(profile)
            if payoffs is None:
                return payoffs
            payoffs = {agent: np.mean(payoff) for agent, payoff in payoffs.items()}
            return payoffs

        # Mixed-strategy profile.
        for agent_id, agent_profile in profile.items():
            # Convert any pure-strategy profiles (int) to a mixtured (list[float]).
            if isinstance(agent_profile, _types.PolicyID):
                new_profile = np.zeros([self.num_policies[agent_id]], dtype=float)
                new_profile[agent_profile] = 1.0
                profile[agent_id] = new_profile

        # Get all possible profiles.
        num_policies = [len(profile[i]) for i in range(self.num_agents)]
        all_profiles = [np.arange(x) for x in num_policies]

        # Get the payoff for each profile.
        total_payoff = {agent_id: 0.0 for agent_id in range(self.num_agents)}
        for possible_profile in itertools.product(*all_profiles):
            possible_profile = {agent_id: int(profile_id) for agent_id, profile_id in enumerate(possible_profile)}

            # Calculate the probability of this pure-profile sampled from the mixture.
            coeffs = []
            for agent_id, policy_id in possible_profile.items():
                coeffs += [profile[agent_id][policy_id]]
            coeff = np.prod(coeffs)
            # If it's unluckily, or not possible, skip it.
            if coeff < 0.0001:
                continue

            payoffs = self.average_payoffs(possible_profile)
            assert payoffs is not None, f"Profile was not simulated: `{possible_profile}`"

            for agent, agent_payoff in payoffs.items():
                total_payoff[agent] += agent_payoff * coeff

        return total_payoff

    def payoff_samples(self, profile: _Profile) -> _ProfilePayoffs:
        """Get the simulated payoffs for a strategy profile.

        Args:
            profile: Map from AgentID to either PolicyID or a mixture over policies.

        Returns:
            Dictionary from AgentID to the list of payoffs for each agent.
        """
        key = self._profile_to_key(profile)
        try:
            return {agent: self.payoffs[key][agent] for agent in range(self.num_agents)}
        except Exception:
            return None

    def num_samples(self, profile: _PureProfile) -> int:
        """Get the number of samples that exist for a particular profile.

        Args:
            profile: PolicyID being played for each Agent.

        Returns:
            Number of samples.
        """
        key = self._profile_to_key(profile)

        if key in self.payoffs:
            return min([len(x) for x in self.payoffs[key].values()])
        else:
            return 0

    def game_matrix(self) -> np.ndarray:
        """Get the matrix-form representation."""
        num_policies = [self.num_policies[i] for i in range(self.num_agents)]
        matrix = np.zeros(num_policies + [self.num_agents], dtype=np.float)
        for profile in itertools.product(*[np.arange(x) for x in num_policies]):
            key = self._profile_to_key({i: pi for i, pi in enumerate(profile)})
            payoffs = self.payoffs[key]
            payoffs = [np.mean(payoffs[i]) for i in range(self.num_agents)]
            matrix[tuple(profile)] = payoffs
        return matrix

    def _profile_to_key(self, profile: _PureProfile) -> str:
        """Get the dict key for the strategy profile."""
        key = ""
        for agent in range(self.num_agents):
            key += f"x{profile[agent]}"
        return key

    def _load_num_policies(self) -> Mapping[_types.PlayerID, int]:
        """Get the number of policies from a loaded shelf."""
        num_policies = {id: 0 for id in range(self.num_agents)}

        for key in self.payoffs.keys():
            policy_ids = key.split("x")[1:]  # Keys are of the form: `x#x#`.
            for player_id, policy_id in enumerate(policy_ids):
                num_policies[player_id] = max(num_policies[player_id], int(policy_id))

        return num_policies

    def __contains__(self, profile) -> bool:
        """Check if this has payoffs for a profile.

        Args:
            profile: profile to check

        Returns:
            true if there is at least one sample of the payoff.
        """
        key = self._profile_to_key(profile)
        return key in self.payoffs
