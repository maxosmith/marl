"""Arena meant for evaluation between agents that are fixed."""
import dataclasses
import operator
from typing import NamedTuple, Optional, Sequence

import numpy as np
import tree
from absl import logging

from marl import types, worlds
from marl.services.arenas import base_arena
from marl.utils import dict_utils, loggers, tree_utils


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

  def run(
      self,
      players,
      *,
      num_episodes: Optional[int] = None,
      num_timesteps: Optional[int] = None,
      **kwargs,
  ) -> Sequence[EpisodeResult]:
    """Run many episodes."""
    del kwargs
    if num_timesteps is not None:
      raise ValueError("SimArena only supports episodic simulation.")
    return [self.run_episode(players) for _ in range(num_episodes)]

  def run_episode(self, players) -> EpisodeResult:
    """Run one episode."""
    timesteps = self.game.reset()
    player_states = {id: None for id in players.keys()}

    has_finished = {id: False for id in players.keys()}
    for player_id, timestep in timesteps.items():
      if timestep.last():
        has_finished[player_id] = True

    # Initialize logging statistics.
    episode_length = 0
    episode_return = {}
    for player_id, player_ts in timesteps.items():
      episode_return[player_id] = player_ts.reward

    while not np.all(list(has_finished.values())):
      # Action selection.
      actions = {}
      for player_id, player_ts in timesteps.items():
        if player_states[player_id] is None:
          player_states[player_id] = players[player_id].episode_reset(player_ts)
        player_states[player_id], actions[player_id] = players[player_id].step(player_states[player_id], player_ts)

      # Transition game state.
      timesteps = self.game.step(actions)
      episode_length += 1

      # Record transition data.
      for player_id, player_ts in timesteps.items():
        if player_id not in episode_return:
          episode_return[player_id] = player_ts.reward
        else:
          episode_return[player_id] = tree.map_structure(operator.iadd, episode_return[player_id], player_ts.reward)
        if player_ts.last():
          has_finished[player_id] = True

    return EpisodeResult(
        episode_length=episode_length,
        episode_return=episode_return,
    )

  def stop(self):
    """Stop running this service if set to run indefinitely."""
    pass
