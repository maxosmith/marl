import dataclasses
import enum
import operator
import warnings
from typing import Mapping, Optional

import numpy as np
import tree
from absl import logging

from marl import _types, individuals, utils, worlds
from marl.services import counter as counter_lib
from marl.services.arenas import base
from marl.utils import dict_utils, loggers, signals, spec_utils, time_utils, tree_utils
from marl.utils.loggers.base import LogData


class _StopwatchKeys(enum.Enum):
    """Keys used by the stopwatch to measure the walltime for subroutines."""

    ACTION = "action"
    TRANSITION = "transition"
    UPDATE = "update"
    STEP = "step"
    EPISODE = "episode"


@dataclasses.dataclass
class EpisodeResult:
    episode_length: int
    episode_return: _types.Tree

    actions_per_second: float
    transitions_per_second: float
    updates_per_second: float
    steps_per_second: float

    def to_logdata(self) -> loggers.LogData:
        """Converts an episode result into data for loggers."""
        log_data = dict(self.__dict__)

        del log_data["episode_return"]
        return_data = tree_utils.flatten_as_dict(self.episode_return)
        return_data = dict_utils.prefix_keys(return_data, "train_return_per_episode/")
        log_data.update(return_data)

        return log_data


@dataclasses.dataclass
class TrainingArena(base.ArenaInterface):
    """Training arenas play games between learning agents and non-learning agents (bots).

    Attributes:
        game:
        players:
        logger:
    """

    game: worlds.Game
    logger: loggers.Logger
    players: Optional[Mapping[_types.PlayerID, individuals.Individual]] = None
    counter: Optional[counter_lib.Counter] = None
    step_key: Optional[str] = None

    def __post_init__(self):
        if self.counter and not self.step_key:
            raise ValueError("Must specify step key with counter.")
        if self.step_key and not self.counter:
            raise ValueError("Must specify counter with step key.")

        self._stop = True

    def run_episode(self) -> EpisodeResult:
        """Run one episode."""

        episode_length = 0
        episode_return = spec_utils.zeros_from_spec(self.game.reward_specs())
        stopwatch = utils.Stopwatch()

        stopwatch.start(_StopwatchKeys.UPDATE.value)

        self._maybe_sychronize_agent_parameters()
        stopwatch.stop(_StopwatchKeys.UPDATE.value)

        timesteps = self.game.reset()
        player_states = {id: player.episode_reset(timesteps[id]) for id, player in self.players.items()}
        while not np.any([ts.last() for ts in timesteps.values()]):
            stopwatch.start(_StopwatchKeys.STEP.value)

            # Action selection.
            stopwatch.start(_StopwatchKeys.ACTION.value)
            actions = {}
            for id, player in self.players.items():
                actions[id], player_states[id] = player.step(timesteps[id], player_states[id])
            stopwatch.stop(_StopwatchKeys.ACTION.value)

            # Environment transition.
            stopwatch.start(_StopwatchKeys.TRANSITION.value)
            timesteps = self.game.step(actions)
            episode_length += 1
            episode_return = tree.map_structure(
                operator.iadd, episode_return, {id: ts.reward for id, ts in timesteps.items()}
            )
            stopwatch.stop(_StopwatchKeys.TRANSITION.value)

            stopwatch.stop(_StopwatchKeys.STEP.value)

        # Inform the agent of the last timestep for logging.
        for id, player in self.players.items():
            player.step(timesteps[id], player_states[id])

        times = stopwatch.get_splits(aggregate_fn=time_utils.mean_per_second)
        return EpisodeResult(
            episode_length=episode_length,
            episode_return=episode_return,
            actions_per_second=times[_StopwatchKeys.ACTION.value],
            transitions_per_second=times[_StopwatchKeys.TRANSITION.value],
            updates_per_second=times[_StopwatchKeys.UPDATE.value],
            steps_per_second=times[_StopwatchKeys.STEP.value],
        )

    def run(
        self,
        num_episodes: Optional[int] = None,
        num_timesteps: Optional[int] = None,
        players: Optional[Mapping[_types.PlayerID, individuals.Individual]] = None,
    ):
        """Runs many episodes serially.

        Runs either `num_episodes` or at least `num_timesteps` timesteps (episodes are run until completion,
        so the true number of timesteps run will include completing the last episode). If neither are arguments
        are specified then this continues to run episodes infinitely.

        Args:
            num_episodes: number of episodes to run.
            num_timesteps: minimum number of timesteps to run.
            players: individuals to play the game. These may be specified at instantiation of the arena for fixed
                players, or dynamically specified through this argument.
        """
        if not (num_episodes is None or num_timesteps is None):
            warnings.warn("Neither `num_episodes` nor `num_timesteps` were specified, running indefinitely.")
        if players and self.players:
            raise ValueError("Cannot override players for an arena with fixed players.")
        if not self.players and not players:
            raise ValueError("Must specify players for an arena without fixed players.")

        logging.info("Running training arena.")

        if players:
            self.players = players

        def _should_terminate(episodes: int, timesteps: int) -> bool:
            episodes_finished = (num_episodes is not None) and (episodes >= num_episodes)
            timesteps_finished = (num_timesteps is not None) and (timesteps >= num_timesteps)
            return episodes_finished or timesteps_finished

        episode_count, timestep_count = 0, 0
        self._stop = False
        stopwatch = utils.Stopwatch(buffer_size=100)

        with signals.runtime_terminator():
            while not _should_terminate(episodes=episode_count, timesteps=timestep_count):
                stopwatch.start(_StopwatchKeys.EPISODE.value)
                result = self.run_episode()
                stopwatch.stop(_StopwatchKeys.EPISODE.value)
                episode_count += 1
                timestep_count += result.episode_length

                logdata = result.to_logdata()
                logdata["episodes_per_second"] = stopwatch.get_splits(aggregate_fn=time_utils.mean_per_second)[
                    _StopwatchKeys.EPISODE.value
                ]
                if self.counter:
                    logdata[self.step_key] = self.counter.get_counts().get(self.step_key, 0)
                self.logger.write(logdata)

                if self._stop:
                    logging.info("Run stopped.")
                    break

        if players:
            self.players = None

    def stop(self):
        """Stop running this service."""
        logging.info("Stopping training arena.")
        self._stop = True

    def _maybe_sychronize_agent_parameters(self):
        for player in self.players.values():
            if isinstance(player, individuals.Agent):
                player.update()
