"""Arena meant for evaluation between agents that are fixed."""
import dataclasses
import operator
from typing import NamedTuple, Sequence

import numpy as np
import tree
from absl import logging

from marl import types, worlds
from marl.services.arenas import base_arena, loggers
from marl.utils import dict_utils, tree_utils


class EpisodeResult(NamedTuple):
    """Data returned from a training episode."""

    episode_length: int
    episode_return: types.Tree

    def to_logdata(self) -> loggers.LogData:
        """Converts an episode result into data for loggers."""
        log_data = dict(self.__dict__)

        del log_data["episode_return"]
        return_data = tree_utils.flatten_as_dict(self.episode_return)
        return_data = dict_utils.prefix_keys(return_data, "episode_return/player_", delimiter="")
        log_data.update(return_data)

        return log_data


@dataclasses.dataclass
class SimArena(base_arena.BaseArena):
    """Simluation arena that play games between individuals."""

    game: worlds.Game

    def simulate_profile(self, profile, num_episodes):
        """Simulate a pure-strategy profile."""
        logging.info("Simulating profile %s for %d episodes.", profile, num_episodes)
        players = {
            player_id: self.strategy_clients[player_id].build_pure_strategy(policy_id)
            for player_id, policy_id in profile.items()
        }
        results = self.run_episodes(players=players, num_episodes=num_episodes)
        return (profile, tree.map_structure(lambda *args: np.stack(args), *results))

    def run(self, players, *, num_episodes) -> Sequence[EpisodeResult]:
        """Run many episodes."""
        return [self.run_episode(players) for _ in range(num_episodes)]

    def run_episode(self, players) -> EpisodeResult:
        """Run one episode."""
        timesteps = self.game.reset()
        player_states = {id: player.episode_reset(timesteps[id]) for id, player in players.items()}

        # Initialize logging statistics.
        episode_length = 0
        episode_return = {id: ts.reward for id, ts in timesteps.items()}

        while not np.all([ts.last() for ts in timesteps.values()]):
            # Action selection.
            actions = {}
            for id, player in players.items():
                player_states[id], actions[id] = player.step(player_states[id], timesteps[id])

            # Transition game state.
            timesteps = self.game.step(actions)
            episode_length += 1
            episode_return = tree.map_structure(
                operator.iadd,
                episode_return,
                {id: ts.reward for id, ts in timesteps.items()},
            )

        return EpisodeResult(
            episode_length=episode_length,
            episode_return=episode_return,
        )

    def stop(self):
        """Stop running this service if set to run indefinitely."""
        pass
