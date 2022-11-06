from typing import Optional

import pyspiel

from marl import worlds
from marl.games import openspiel_proxy
from marl_experiments.roshambo import roshambo_bot

GAME_STRING = "repeated_game(stage_game=matrix_rps(),num_repetitions=1000)"


def build_game(num_throws: Optional[int] = roshambo_bot.ROSHAMBO_NUM_THROWS) -> worlds.Game:
    """Build an instance of Repeated-RPS that is compatible with the bot population."""
    # NOTE: pyspiel bots require the full state.
    env = openspiel_proxy.OpenSpielProxy(
        f"repeated_game(stage_game=matrix_rps(),num_repetitions={pyspiel.ROSHAMBO_NUM_THROWS})",
        include_full_state=True,
    )
    return env
