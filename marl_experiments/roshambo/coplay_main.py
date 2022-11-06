"""Simulate a coplay matrix for the Roshambo bots."""
import itertools

import numpy as np
import tree
from absl import app, logging

from marl_experiments.roshambo import roshambo_bot
from marl_experiments.roshambo.services import evaluation_arena
from marl_experiments.roshambo.utils import utils as rps_utils

_NUM_EPISODES = 10


def main(_):
    """Simulate a coplay matrix for the Roshambo bots."""
    game = rps_utils.build_game()
    arena = evaluation_arena.EvaluationArena()

    bot_names = roshambo_bot.ROSHAMBO_BOT_NAMES[:3]
    for name0, name1 in itertools.product(bot_names, repeat=2):
        players = {0: roshambo_bot.RoshamboBot(name0), 1: roshambo_bot.RoshamboBot(name1)}
        results = arena.run_episodes(game=game, players=players, num_episodes=_NUM_EPISODES)
        results = tree.map_structure(lambda *x: np.mean([x]), *results)
        logging.info(f"{name0} vs {name1}: {results}")


if __name__ == "__main__":
    app.run(main)
