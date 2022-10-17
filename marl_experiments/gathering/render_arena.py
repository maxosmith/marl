"""Arena meant for evaluation between agents that are not learning (actively)."""
import dataclasses
import operator
import os.path as osp
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Union
from unittest import result

import numpy as np
import tree
from absl import logging
from PIL import Image

from marl import _types, services, utils, worlds
from marl.services.arenas import base
from marl.utils import dict_utils, loggers, signals, spec_utils, tree_utils


@dataclasses.dataclass
class EpisodeResult:
    episode_length: int
    episode_return: _types.Tree
    render: Sequence

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
    name: Optional[str] = None
    game_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    agent_id_to_player_id: Optional[Mapping[_types.PlayerID, _types.PlayerID]] = None
    bot_id_to_player_id: Optional[Mapping[_types.PlayerID, _types.PlayerID]] = None

    def __post_init__(self):
        # If PlayerID mappings were not specified, we assume that the PlayerID of the
        # agents/bots in their populations are unique and correct.
        if not self.agent_id_to_player_id:
            self.agent_id_to_player_id = lambda x: x
        if not self.bot_id_to_player_id:
            self.bot_id_to_player_id = lambda x: x


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
        counter: services.Counter,
        step_key: str,
        result_dir: utils.ResultDirectory,
    ):
        self._agents = agents
        self._bots = bots
        if isinstance(scenarios, EvaluationScenario):
            scenarios = [scenarios]
        self._scenarios = scenarios
        self._evaluation_frequency = evaluation_frequency
        self._counter = counter
        self._last_eval = -np.inf
        self._step_key = step_key
        self._result_dir = result_dir

    def run_episode(self, game: worlds.Game, players: Mapping[_types.PlayerID, _types.Individual]) -> EpisodeResult:
        """Run one episode."""
        episode_length = 0
        episode_return = spec_utils.zeros_from_spec(game.reward_specs())

        timesteps = game.reset()
        player_states = {id: player.episode_reset(timesteps[id]) for id, player in players.items()}
        images = [game.render(mode="image")]

        while not np.any([ts.last() for ts in timesteps.values()]):
            actions = {}
            for id, player in players.items():
                actions[id], player_states[id] = player.step(timesteps[id], player_states[id])

            timesteps = game.step(actions)
            images.append(game.render(mode="image"))

            episode_length += 1
            episode_return = tree.map_structure(
                operator.iadd, episode_return, {id: ts.reward for id, ts in timesteps.items()}
            )

        return EpisodeResult(episode_length=episode_length, episode_return=episode_return, render=images)

    def run_evaluation(self, step: int):
        logging.info("Running evaluation scenarios.")
        result_dir: str = self._result_dir.make_subdir(f"step_{step}").dir

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

            # Save the render to disk.
            for result_i, result in enumerate(results):
                images = result.render
                save_path = osp.join(
                    result_dir, f"{scenario.name}_{result_i}.gif" if scenario.name else f"{result_i}.gif"
                )
                images[0].save(save_path, save_all=True, append_images=images[1:], duration=100, loop=0, quality=100)

        logging.info("Evaluation complete.")

    def run(self):
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
