"""Simulate a coplay matrix for the Roshambo bots."""
import itertools
import multiprocessing
import os.path as osp
import pickle
from typing import Any, Mapping, NamedTuple, Tuple

import numpy as np
import tree
from absl import app, flags, logging

from marl import _types, worlds
from marl.games import openspiel_proxy
from marl_experiments.roshambo import roshambo_bot
from marl_experiments.roshambo.services import evaluation_arena
from marl_experiments.roshambo.utils import utils as rps_utils

_RESULT_DIR = "/scratch/wellman_root/wellman1/mxsmith/data/roshambo/coplay"
_NUM_EPISODES = 100


flags.DEFINE_integer("num_workers", 7, "Number of worker processes.")
FLAGS = flags.FLAGS


class Demonstration(NamedTuple):
    """Demonstration of an episode."""

    observation: _types.Array
    actions: _types.Array
    rewards: _types.Array
    step_type: _types.Array


class DemonstrationArena(evaluation_arena.EvaluationArena):
    """Arena for generating a demonstration dataset for behavioural cloning."""

    def run_episode(self, game: worlds.Game, players: Mapping[_types.PlayerID, _types.Individual]) -> Any:
        """Run one episode.

        NOTE: This assumes that all players are RPS bots, and can accept the serialized state as input.
        """
        demonstration = Demonstration(
            observation=np.zeros((1001, 6), dtype=np.float32),
            actions=np.zeros((1001, 2), dtype=np.int32),
            rewards=np.zeros((1001, 2), dtype=np.float32),
            step_type=np.zeros((1001,), dtype=np.int32),
        )
        t = 0

        timesteps = game.reset()
        player_states = {
            id: player.episode_reset(timestep=timesteps[id], player_id=id) for id, player in players.items()
        }

        while not np.any([ts.last() for ts in timesteps.values()]):
            actions = {}
            for id, player in players.items():
                actions[id], player_states[id] = player.step(timesteps[id], player_states[id])

            demonstration.observation[t, :] = timesteps[0].observation[openspiel_proxy.INFO_STATE]
            demonstration.step_type[t] = int(timesteps[0].step_type)
            demonstration.actions[t, 0] = actions[0]
            demonstration.actions[t, 1] = actions[1]

            timesteps = game.step(actions)

            demonstration.rewards[t, 0] = timesteps[0].reward
            demonstration.rewards[t, 1] = timesteps[1].reward
            t += 1

        demonstration.observation[t, :] = timesteps[0].observation[openspiel_proxy.INFO_STATE]
        demonstration.step_type[t] = int(timesteps[0].step_type)

        return demonstration


def worker(payload: Tuple[str, str]) -> None:
    """Worker that generates coplay data for a single strategy profile."""
    name0, name1 = payload

    game = rps_utils.build_game()
    arena = DemonstrationArena()
    players = {0: roshambo_bot.RoshamboBot(name0), 1: roshambo_bot.RoshamboBot(name1)}

    results = arena.run_episodes(game=game, players=players, num_episodes=_NUM_EPISODES)
    results = tree.map_structure(lambda *x: np.stack(x), *results)
    save_path = osp.join(_RESULT_DIR, f"{name0}_vs_{name1}.pb")
    pickle.dump(results, open(save_path, "wb"))

    return payload


def main(_):
    """Simulate a coplay matrix for the Roshambo bots."""
    bot_names = roshambo_bot.ROSHAMBO_BOT_NAMES

    with multiprocessing.Pool(processes=FLAGS.num_workers) as pool:
        jobs = [(name0, name1) for name0, name1 in itertools.combinations_with_replacement(bot_names, 2)]
        results = pool.map(worker, jobs)

        logging.info("Finished profiles:")
        for name0, name1 in results:
            logging.info("%s vs. %s", name0, name1)


if __name__ == "__main__":
    app.run(main)
