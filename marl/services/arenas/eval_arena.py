"""Interface for servies that simulate episodes."""
import dataclasses
import time
from typing import Any, Callable, Mapping, NamedTuple, Optional, Sequence, Union

import numpy as np
import tree
from absl import logging

from marl import individuals, types, worlds
from marl.services import snapshotter as snap_lib
from marl.services.arenas import base_arena
from marl.utils import dict_utils, loggers, signals, tree_utils


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
class EvaluationScenario:
  """A scenario (setting) to evaluate all players against."""

  game_ctor: Callable[..., worlds.Game]
  num_episodes: int
  game_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
  name: Optional[str] = None
  aggregate_fn: Any = np.mean


@dataclasses.dataclass
class EvalArena(base_arena.BaseArena):
  """Arena for periodic evaluation of a learner.

  Args:
    snapshotter: Service for saving an evaluated learner's parameters.
    snapshotter_scenario_id: Scenario ID used to measure snapshot priorities.
    learner_id: Player ID to snapshot.
  """

  players: Mapping[types.PlayerID, individuals.Bot | individuals.Agent]
  scenarios: Union[EvaluationScenario, Sequence[EvaluationScenario]]
  counter: Any
  step_key: str
  evaluation_frequency: int = 1
  logger: Any | None = None
  snapshotter: snap_lib.PrioritySnapshotter | None = None
  snapshotter_scenario_id: int | None = None
  learner_id: int | None = None

  def __post_init__(self):
    """Post initializer."""
    if isinstance(self.scenarios, EvaluationScenario):
      self.scenarios = [self.scenarios]
    self._running = False
    self._stop = False
    self._last_eval = -1

  def run_evaluation(self, step: Optional[int] = None):
    logging.info("Running evaluation scenarios.")
    # Get the current state of all learning agents.
    for agent in self.players.values():
      agent.update()

    evaluation_results = []

    for scenario_i, scenario in enumerate(self.scenarios):
      if scenario.name:
        logging.info("\tRunning scenario: %s", scenario.name)

      game = scenario.game_ctor(**scenario.game_kwargs)

      results = [self._run_episode(game, self.players) for _ in range(scenario.num_episodes)]
      results = tree.map_structure(lambda *args, scenario=scenario: scenario.aggregate_fn([args]), *results)
      evaluation_results.append(results)

      if self.logger:
        results = results.to_logdata()
        if step:
          results[self.step_key] = step
        if scenario.name:
          results = dict_utils.prefix_keys(results, scenario.name)
        self.logger.write(results)

      if self.snapshotter and (scenario_i == self.snapshotter_scenario_id):
        self.snapshotter.save(results[f"episode_return/player_{self.learner_id}"], self.players[self.learner_id].params)

    logging.info("Evaluation complete.")
    return evaluation_results

  def _run_episode(self, game, players) -> EpisodeResult:
    """Run a single evaluation episode."""
    timesteps = game.reset()
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

      # Transition game state.
      timesteps = game.step(actions)
      episode_length += 1
      for player_id, timestep in timesteps.items():
        episode_return[player_id] += timestep.reward
        if timestep.last():
          has_finished[player_id] = True

    return EpisodeResult(
        episode_length=episode_length,
        episode_return=episode_return,
    )

  def run(self, *, num_episodes: Optional[int] = None, num_timesteps: Optional[int] = None, **kwargs):
    """Periodically run all evaluation scenarios."""
    del num_episodes, num_timesteps, kwargs
    self._running = True
    with signals.runtime_terminator():
      while True:
        if self._stop:  # Check if the services was manually stopped.
          self._stop = False
          break

        # Check the current step.
        counts = self.counter.get_counts()
        step = counts.get(self.step_key, 0)

        if step >= (self._last_eval + self.evaluation_frequency):
          self.run_evaluation(step)
          self._last_eval = step

        # Don't spam the counter.
        for _ in range(10):
          # Do not sleep for a long period of time to avoid LaunchPad program
          # termination hangs (time.sleep is not interruptible).
          time.sleep(1)
    self._running = False

  def stop(self):
    """Stop running this service if set to run indefinitely."""
    if not self._running:
      raise RuntimeError("Tried to stop an arena that isn't running.")
    self._stop = True
