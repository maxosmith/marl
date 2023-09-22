"""Interface for servies that simulate episodes."""
import dataclasses
import itertools
import operator
from typing import Any, Callable, NamedTuple, Optional, Sequence

import numpy as np
import tree
from absl import logging

from marl import types
from marl.services.arenas import base_arena
from marl.utils import dict_utils, loggers, tree_utils

_StopFn = Callable[[int, int], bool]


class EpisodeResult(NamedTuple):
  """Data returned from a training episode."""

  episode_length: int
  episode_return: types.Tree

  def to_logdata(self) -> loggers.LogData:
    """Converts an episode result into data for loggers."""
    log_data = dict(episode_length=self.episode_length)

    return_data = tree_utils.flatten_as_dict(self.episode_return)
    return_data = dict_utils.prefix_keys(return_data, "episode_return/player_", delimiter="")
    log_data.update(return_data)

    return log_data


@dataclasses.dataclass
class TrainArena(base_arena.BaseArena):
  """Interface for an agent-environment interaction (arena) services.

  This assumes that only a single agent is learning against a fixed set of coplayers.

  Notes:
    - Player's parameters are assumed to synced at the end of episodes through `episode_reset`.
    - Per-timestep syncs are performed through `update`, which the learner should customize to
      suit its required degree of on-policy-ness.
  """

  game: Any
  adder: Any
  logger: Optional[Any] = None
  counter: Optional[Any] = None
  step_key: Optional[str] = None

  def __post_init__(self):
    if self.counter and not self.step_key:
      raise ValueError("Must specify step key with counter.")
    if self.step_key and not self.counter:
      raise ValueError("Must specify counter with step key.")

    self._running = False

  def run(
      self,
      learner_id: types.PlayerID,
      players: Any,
      *,
      num_episodes: Optional[int] = None,
      num_timesteps: Optional[int] = None,
  ) -> Sequence[EpisodeResult]:
    """Run the arena to generate experience.

    Runs either `num_episodes`, or _at least_ `num_timesteps`, or indefinitely.

    Args:
      num_episodes: Number of episodes to run.
      num_timesteps: Minimum number of timesteps to run. It will let the last episode
        complete.
    """
    if self._running:
      raise RuntimeError("Tried to run an already running arena.")
    self._running = True

    if (num_timesteps is None) == (num_episodes is None):
      print(num_timesteps is None, num_episodes is None)
      logging.info("One of `num_episodes`, `num_timesteps` must be specified.")
    elif num_episodes is not None:
      logging.info(f"Running for {num_episodes=}.")
    else:
      logging.info(f"Running for {num_timesteps=}.")

    def _should_stop(episodes: int, timesteps: int) -> bool:
      """Checks if the arena should stop generating experience."""
      episodes_finished = (num_episodes is not None) and (episodes >= num_episodes)
      timesteps_finished = (num_timesteps is not None) and (timesteps >= num_timesteps)
      return episodes_finished or timesteps_finished

    self._run(learner_id, players, _should_stop)

    self._running = False

  def _run(self, learner_id: types.PlayerID, players: Any, should_stop: _StopFn):
    """Agent-environment action loop."""
    total_timesteps = 0
    for episode_i in itertools.count():
      results = self._run_episode(learner_id=learner_id, players=players)
      total_timesteps += results.episode_length

      # Maybe write episodic results to log.
      logdata = results.to_logdata()
      if self.counter:
        logdata[self.step_key] = self.counter.get_counts().get(self.step_key, 0)
      if self.logger:
        self.logger.write(logdata)

      if should_stop(episode_i + 1, total_timesteps):
        break

  def _run_episode(self, learner_id, players) -> EpisodeResult:
    timesteps = self.game.reset()
    # Reset player's episodic state, and likely force sync their parameters.
    player_states = {id: player.episode_reset(timesteps[id]) for id, player in players.items()}

    # Initialize logging statistics.
    episode_length = 0
    episode_return = {id: ts.reward for id, ts in timesteps.items()}

    while not np.all([ts.last() for ts in timesteps.values()]):
      # Action selection.
      actions = {}
      for id, player in players.items():
        player_states[id], actions[id] = player.step(player_states[id], timesteps[id])
      self.adder.add(
          timestep=timesteps[learner_id],
          action=actions[learner_id],
          extras=player_states[learner_id],
      )

      # Transition game state.
      timesteps = self.game.step(actions)
      episode_length += 1
      episode_return = tree.map_structure(
          operator.iadd,
          episode_return,
          {id: ts.reward for id, ts in timesteps.items()},
      )

    # Log final timestep.
    # TODO(max): Replace dummy action/extras with zeros-like of the trees.
    self.adder.add(
        timestep=timesteps[learner_id],
        action=actions[learner_id],
        extras=player_states[learner_id],
    )

    return EpisodeResult(
        episode_length=episode_length,
        episode_return=episode_return,
    )

  def stop(self):
    """Stop running this service if set to run indefinitely."""
    if not self._running:
      raise RuntimeError("Tried to stop an arena that isn't running.")
