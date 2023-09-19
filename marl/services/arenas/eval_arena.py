"""Interface for servies that simulate episodes."""
import dataclasses
import itertools
import operator
from typing import Any, Callable, Mapping, NamedTuple, Optional, Sequence, Union

import numpy as np
import tree
from absl import logging

from marl import types, worlds
from marl.services.arenas import base_arena, loggers
from marl.utils import dict_utils, tree_utils

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
class EvaluationScenario:
    """A scenario (setting) to evaluate all players against."""

    game_ctor: Callable[..., worlds.Game]
    num_episodes: int
    game_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    name: Optional[str] = None
    aggregate_fn: Any = np.mean


@dataclasses.dataclass
class EvalArena(base_arena.BaseArena):
    """Arena for periodic evaluation of a learner."""

    scenarios: Union[EvaluationScenario, Sequence[EvaluationScenario]]
    logger: Optional[Any] = None
    counter: Optional[Any] = None
    step_key: Optional[str] = None

    def __post_init__(self):
        if self.counter and not self.step_key:
            raise ValueError("Must specify step key with counter.")
        if self.step_key and not self.counter:
            raise ValueError("Must specify counter with step key.")

        if isinstance(self.scenarios, EvaluationScenario):
            self.scenarios = [self.scenarios]

        self._running = False

    def run_evaluation(self, players: Any, step: Optional[int] = None):
        logging.info("Running evaluation scenarios.")
        # Get the current state of all learning agents.
        for agent in players.values():
            agent.update()

        evaluation_results = []

        for scenario in self.scenarios:
            if scenario.name:
                logging.info(f"\tRunning scenario: {scenario.name}")

            game = scenario.game_ctor(**scenario.game_kwargs)

            results = [self._run_episode(game, players) for _ in range(scenario.num_episodes)]
            results = tree.map_structure(lambda *args: scenario.aggregate_fn([args]), *results)
            evaluation_results.append(results)

            if self.logger:
                results = results.to_logdata()
                if step:
                    results[self._step_key] = step
                if scenario.name:
                    results = dict_utils.prefix_keys(results, scenario.name)
                self.logger.write(results)

        logging.info("Evaluation complete.")
        return evaluation_results

    def _run_episode(self, game, players) -> EpisodeResult:
        timesteps = game.reset()
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

            # Transition game state.
            timesteps = game.step(actions)
            episode_length += 1
            episode_return = tree.map_structure(
                operator.iadd,
                episode_return,
                {id: ts.reward for id, ts in timesteps.items()},
            )

            # Maybe sync agent parameters.
            for player in players.values():
                player.update()

        return EpisodeResult(
            episode_length=episode_length,
            episode_return=episode_return,
        )

    def run(
        self,
        *,
        num_episodes: Optional[int] = None,
        num_timesteps: Optional[int] = None,
        **kwargs,
    ):
        """Run the arena to generate experience.

        Runs either `num_episodes`, or _at least_ `num_timesteps`, or indefinitely.

        Args:
          num_episodes: Number of episodes to run.
          num_timesteps: Minimum number of timesteps to run.
        """
        pass

    def stop(self):
        """Stop running this service if set to run indefinitely."""
        if not self._running:
            raise RuntimeError("Tried to stop an arena that isn't running.")
