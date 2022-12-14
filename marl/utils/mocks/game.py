from typing import Optional, Sequence

import numpy as np

from marl import _types, worlds
from marl.utils import spec_utils
from marl.utils.mocks import environment as mock_envs


class Game(worlds.Game):
    """A mock game that generates placeholder values according to a prespecified spec."""

    def __init__(self, player_to_spec: worlds.GameSpec, *, episode_length: int = 25):
        self._player_to_mock_env = {
            id: mock_envs.Environment(spec, episode_length=episode_length) for id, spec in player_to_spec.items()
        }

    def step(self, actions: _types.PlayerIDToAction) -> worlds.TimeStep:
        for player_id, mock_env in self._player_to_mock_env.items():
            spec_utils.validate_spec(mock_env.action_spec(), actions[player_id])
        return {id: env.step(actions[id]) for id, env in self._player_to_mock_env.items()}

    def reset(self) -> worlds.TimeStep:
        return {id: env.reset() for id, env in self._player_to_mock_env.items()}

    def action_specs(self) -> worlds.PlayerIDToSpec:
        return {id: env.action_spec() for id, env in self._player_to_mock_env.items()}

    def observation_specs(self) -> worlds.PlayerIDToSpec:
        return {id: env.observation_spec() for id, env in self._player_to_mock_env.items()}

    def reward_specs(self) -> worlds.PlayerIDToSpec:
        return {id: env.reward_spec() for id, env in self._player_to_mock_env.items()}


class SymmetricSpecGame(Game):
    def __init__(self, env_spec: worlds.EnvironmentSpec, *, num_players: int, episode_length: int = 25):
        Game.__init__(self, {id: env_spec for id in range(num_players)}, episode_length=episode_length)


class DiscreteSymmetricSpecGame(Game):
    def __init__(
        self,
        *,
        num_players: int,
        num_actions: int = 1,
        observation_shape: Sequence[int] = (),
        reward_spec: Optional[worlds.TreeSpec] = None,
        episode_length: int = 25,
    ):
        # Default reward is a single continuous scalar.
        if reward_spec is None:
            reward_spec = worlds.ArraySpec((), np.float32)

        observation_spec = worlds.ArraySpec(shape=observation_shape, dtype=np.int32)
        action_spec = worlds.DiscreteArraySpec(num_actions, dtype=np.int32)

        env_spec = worlds.EnvironmentSpec(
            action=action_spec,
            observation=observation_spec,
            reward=reward_spec,
        )
        SymmetricSpecGame.__init__(self, env_spec=env_spec, num_players=num_players, episode_length=episode_length)
