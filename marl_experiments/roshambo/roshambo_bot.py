"""OpenSpiel Bot proxy and utilities."""
from typing import Optional, Sequence, Tuple

import pyspiel

from marl import _types, individuals, worlds

ROSHAMBO_BOT_NAMES: Sequence[str] = pyspiel.roshambo_bot_names()
ROSHAMBO_BOT_NAMES.sort()

ROSHAMBO_NUM_THROWS = pyspiel.ROSHAMBO_NUM_THROWS


class RoshamboBot(individuals.Bot):
    """OpenSpiel RPS bot.

    These bots require a custom interface, because:
        (a) They require access to the underlying game state implemented in OpenSpiel.
        (b) They must be notified of their player ID for each episode (if it changes).
    """

    def __init__(self, name: str, num_throws: Optional[int] = ROSHAMBO_NUM_THROWS):
        """Initializes an instance of `RoshamboBot`."""
        if name not in ROSHAMBO_BOT_NAMES:
            raise ValueError(f"Unknown bot `{name}', expected one of: {ROSHAMBO_BOT_NAMES}.")

        self._name = name
        self._num_throws = num_throws
        self._bot = None
        self._previous_id = None

    def step(
        self, timestep: worlds.TimeStep, state: Optional[_types.State] = None
    ) -> Tuple[_types.Action, _types.State]:
        """Samples an action from the underlying bot."""
        assert self._bot, "Episode reset must called before step to tell the bot what ID it is playing."

        _, game_state = pyspiel.deserialize_game_and_state(timestep.observation["serialized_state"])
        action = self._bot.step(game_state)

        return action, state

    def episode_reset(self, player_id: int, timestep: worlds.TimeStep) -> _types.State:
        """Resets the bot, informing of it the player ID for the next episode."""
        if player_id == self._previous_id:
            # If the player ID is the same, we do not need to rebuild the bot.
            self._bot.restart()
            return

        self._bot = pyspiel.make_roshambo_bot(player_id, self._name)
        self._previous_id = player_id
        return None
