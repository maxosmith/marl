"""Interface for servies that simulate episodes."""
import dataclasses
import itertools
from typing import Any, Callable, NamedTuple, Optional, Sequence

import numpy as np
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
  learner_id: types.PlayerID
  players: Any
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
    self._stop = False

  def run(
      self,
      *,
      num_episodes: Optional[int] = None,
      num_timesteps: Optional[int] = None,
      **kwargs,
  ) -> Sequence[EpisodeResult]:
    """Run the arena to generate experience.

    Runs either `num_episodes`, or _at least_ `num_timesteps`, or indefinitely.

    Args:
      num_episodes: Number of episodes to run.
      num_timesteps: Minimum number of timesteps to run. It will let the last episode
        complete.
    """
    del kwargs
    if self._running:
      raise RuntimeError("Tried to run an already running arena.")
    self._running = True

    if num_episodes is not None:
      logging.info("Running for num_episodes=%d.", num_episodes)
    elif num_timesteps is not None:
      logging.info("Running for num_timesteps=%d.", num_timesteps)
    else:
      logging.info("Running indefinitely.")

    def _should_stop(episodes: int, timesteps: int) -> bool:
      """Checks if the arena should stop generating experience."""
      if (num_episodes is None) and (num_timesteps is None):
        return False
      episodes_finished = (num_episodes is not None) and (episodes >= num_episodes)
      timesteps_finished = (num_timesteps is not None) and (timesteps >= num_timesteps)
      return episodes_finished or timesteps_finished

    self._run(self.learner_id, self.players, _should_stop)

    self._running = False

  def _run(self, learner_id: types.PlayerID, players: Any, should_stop: _StopFn):
    """Agent-environment action loop."""
    total_timesteps = 0
    for episode_i in itertools.count():
      if self._stop:  # Check if the services was manually stopped.
        self._stop = False
        break

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
    player_states = {id: None for id in players.keys()}

    has_finished = {id: False for id in players.keys()}
    for player_id, timestep in timesteps.items():
      if timestep.last():
        has_finished[player_id] = True

    # Initialize logging statistics.
    episode_length = 0
    episode_return = {id: 0.0 for id in players.keys()}
    for player_id, timestep in timesteps.items():
      episode_return[player_id] += timestep.reward

    while not np.all(list(has_finished.values())):
      # Action selection.
      actions = {}
      for player_id, timestep in timesteps.items():
        if player_states[player_id] is None:
          # Player's first timetep: episodic state, and likely force sync their parameters.
          player_states[player_id] = players[player_id].episode_reset(timestep)
        player_states[player_id], actions[player_id] = players[player_id].step(player_states[player_id], timestep)

      # Maybe log the learner's experience.
      if learner_id in timesteps:
        self.adder.add(
            timestep=timesteps[learner_id],
            action=actions[learner_id],
            extras=player_states[learner_id],
        )

      # Transition game state.
      timesteps = self.game.step(actions)
      episode_length += 1
      for player_id, timestep in timesteps.items():
        episode_return[player_id] += timestep.reward
        if timestep.last():
          has_finished[player_id] = True

    # Log final timestep.
    if learner_id in timesteps:
      # TODO(max): Replace dummy action/extras with zeros-like of the trees.
      self.adder.add(
          timestep=timesteps[learner_id],
          action=self.game.action_specs()[learner_id].generate_value(),
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
    self._stop = True
