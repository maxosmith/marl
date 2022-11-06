"""Arena meant for evaluation between agents that are not learning (actively)."""
import dataclasses
import operator
import time
from typing import Any, Callable, Mapping, NamedTuple, Optional, Sequence, Union

import numpy as np
import tree
from absl import logging

from marl import _types, services, worlds
from marl.services import counter as counter_lib
from marl.services.arenas import base
from marl.utils import dict_utils, loggers, signals, spec_utils, tree_utils


class EpisodeResult(NamedTuple):
    """Collection of metrics summarizing an episode."""

    episode_length: np.array
    episode_return: _types.Tree

    def to_logdata(self) -> loggers.LogData:
        """Converts an episode result into data for loggers."""
        log_data = dict(episode_length=self.episode_length)

        return_data = tree_utils.flatten_as_dict(self.episode_return)
        return_data = dict_utils.prefix_keys(return_data, "eval_return/")
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
        """Post-initializer for EvaluationScenario."""
        # If PlayerID mappings were not specified, we assume that the PlayerID of the
        # agents/bots in their populations are unique and correct.
        if not self.agent_id_to_player_id:
            self.agent_id_to_player_id = lambda x: x
        if not self.bot_id_to_player_id:
            self.bot_id_to_player_id = lambda x: x


class EvaluationArena(base.ArenaInterface):
    """Evaluation arenas play games between individuals."""

    def __init__(
        self,
        agents: Mapping[_types.PlayerID, services.LearnerPolicy] = {},
        bots: Mapping[_types.PlayerID, _types.Individual] = {},
        scenarios: Union[EvaluationScenario, Sequence[EvaluationScenario]] = [],
        evaluation_frequency: Optional[int] = None,
        counter: Optional[counter_lib.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        step_key: Optional[str] = None,
    ):
        """Initializes an instance of an EvaluationArena."""
        self._agents = agents
        self._bots = bots
        if isinstance(scenarios, EvaluationScenario):
            scenarios = [scenarios]
        self._scenarios = scenarios
        self._evaluation_frequency = evaluation_frequency
        self._counter = counter
        self._logger = logger
        self._step_key = step_key
        self._last_eval = -np.inf

    def run_episode(self, game: worlds.Game, players: Mapping[_types.PlayerID, _types.Individual]) -> EpisodeResult:
        """Run one episode."""
        episode_length = 0
        episode_return = spec_utils.zeros_from_spec(game.reward_specs())

        timesteps = game.reset()
        player_states = {
            id: player.episode_reset(timestep=timesteps[id], player_id=id) for id, player in players.items()
        }

        while not np.any([ts.last() for ts in timesteps.values()]):
            actions = {}
            for id, player in players.items():
                actions[id], player_states[id] = player.step(timesteps[id], player_states[id])

            timesteps = game.step(actions)

            episode_length += 1
            episode_return = tree.map_structure(
                operator.iadd, episode_return, {id: ts.reward for id, ts in timesteps.items()}
            )

        return EpisodeResult(episode_length=np.asarray(episode_length), episode_return=episode_return)

    def run_evaluation(self, step: int):
        """Run evaluation of all scenarios."""
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
                players[scenario.agent_id_to_player_id(agent_id)] = agent
            for bot_id, bot in self._bots.items():
                players[scenario.bot_id_to_player_id(bot_id)] = bot

            results = [self.run_episode(game, players) for _ in range(scenario.num_episodes)]
            results = tree.map_structure(lambda *args: np.mean([args]), *results)
            results = results.to_logdata()
            results[self._step_key] = step
            if scenario.name:
                results = dict_utils.prefix_keys(results, scenario.name)
            self._logger.write(results)

        logging.info("Evaluation complete.")

    def run(self):
        """Periodically run evaluation until program termination."""
        with signals.runtime_terminator():
            while True:
                # Check the current step.
                counts = self._counter.get_counts()
                step = counts.get(self._step_key, 0)

                if step >= (self._last_eval + self._evaluation_frequency):
                    self.run_evaluation(step)
                    self._last_eval = step

                # Don't spam the counter.
                for _ in range(10):
                    # Do not sleep for a long period of time to avoid LaunchPad program
                    # termination hangs (time.sleep is not interruptible).
                    time.sleep(1)
