"""Arena meant for evaluation between agents that are not learning (actively)."""
import dataclasses
import operator
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import numpy as np
import tree
from absl import logging

from marl import _types, services, worlds
from marl.services.arenas import base
from marl.utils import dict_utils, loggers, signals, spec_utils, tree_utils


@dataclasses.dataclass
class EpisodeResult:
    episode_length: int
    episode_return: _types.Tree

    def to_logdata(self) -> loggers.LogData:
        """Converts an episode result into data for loggers."""
        log_data = dict(self.__dict__)

        del log_data["episode_return"]
        return_data = tree_utils.flatten_as_dict(self.episode_return)
        return_data = dict_utils.prefix_keys(return_data, "episode_return/")
        log_data.update(return_data)

        return log_data


@dataclasses.dataclass
class EvaluationScenario:
    """A scenario (setting) to evaluate all players against."""

    game_ctor: Callable[..., worlds.Game]
    num_episodes: int
    game_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    name: Optional[str] = None
    agent_id_to_player_id: Optional[Mapping[_types.PlayerID, _types.PlayerID]] = None
    bot_id_to_player_id: Optional[Mapping[_types.PlayerID, _types.PlayerID]] = None

    def __post_init__(self):
        if self.agent_id_to_player_id and (not self.bot_id_to_player_id):
            raise ValueError("Must specify both ID mappings if either is specified.")
        if self.bot_id_to_player_id and (self.agent_id_to_player_id):
            raise ValueError("Must specify both ID mappings if either is specified.")
        if (not self.agent_id_to_player_id) and (not self.bot_id_to_player_id):
            return

        # If neither PlayerID mappings were specified, we assume that the PlayerID of the
        # agents/bots in their populations are unique and correct.
        self.agent_id_to_player_id = lambda x: x
        self.bot_id_to_player_id: lambda x: x


class EvaluationArena(base.ArenaInterface):
    """Evaluation arenas play games between individuals.

    Args:
        agents:
        bots:
        scenarios:
        evaluation_frequncy: Frequency, in seconds, to run evaluation.
        logger:
    """

    def __init__(
        self,
        agents: Mapping[_types.PlayerID, services.LearnerPolicy],
        bots: Mapping[_types.PlayerID, _types.Individual],
        scenarios: Union[EvaluationScenario, Sequence[EvaluationScenario]],
        evaluation_frequency: int,
        logger: loggers.Logger,
    ):
        self._agents = agents
        self._bots = bots
        if isinstance(scenarios, EvaluationScenario):
            scenarios = [scenarios]
        self._scenarios = scenarios
        self._evaluation_frequency = evaluation_frequency
        self._logger = logger

    def run_episode(self, game: worlds.Game, players: Mapping[_types.PlayerID, _types.Individual]) -> EpisodeResult:
        """Run one episode."""
        episode_length = 0
        episode_return = spec_utils.zeros_from_spec(game.reward_specs())

        timesteps = game.reset()
        player_states = {id: player.episode_reset(timesteps[id]) for id, player in players.items()}

        while not np.any([ts.last() for ts in timesteps.values()]):
            actions = {}
            for id, player in players.items():
                actions[id], player_states[id] = player.step(timesteps[id], player_states[id])

            timesteps = game.step(actions)

            episode_length += 1
            episode_return = tree.map_structure(
                operator.iadd, episode_return, {id: ts.reward for id, ts in timesteps.items()}
            )

        return EpisodeResult(episode_length=episode_length, episode_return=episode_return)

    def run_evaluation(self):
        logging.info("Running evaluation scenarios.")

        # Get the current state of all learning agents.
        for agent in self._agents.values():
            agent.sync_params()

        for scenario in self._scenarios:
            if scenario.name:
                logging.info(f"\tRunning scenario: {scenario.name}")

            game = scenario.game_ctor(**scenario.game_kwargs)
            players = {}
            for agent_id, agent in self._agents.items():
                players[scenario.agent_id_to_player_id[agent_id]] = agent
            for bot_id, bot in self._bots.items():
                players[scenario.bot_id_to_player_id[bot_id]] = bot

            results = [self.run_episode(game, players) for _ in range(scenario.num_episodes)]
            results = tree.map_structure(np.mean, *results)
            results = results.to_logdata()
            if scenario.name:
                results = dict_utils.prefix_keys(results, scenario.name)
            self._logger.write(results)

        logging.info("Evaluation complete.")

    def run(self):
        with signals.runtime_terminator():
            while True:
                self.run_evaluation()

                # Wait for a set period before re-evaluating.
                for _ in range(self._evaluation_frequency):
                    # Do not sleep for a long period of time to avoid LaunchPad program
                    # termination hangs (time.sleep is not interruptible).
                    time.sleep(1)
