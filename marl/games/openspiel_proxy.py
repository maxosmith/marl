"""Proxy for OpenSpiel games."""
from typing import Union

import numpy as np
import pyspiel
from absl import logging
from open_spiel.python import rl_environment
from pyspiel import PlayerId as TurnId

from marl import _types, worlds

_CURRENT_PLAYER = "current_player"
SERIALIZED_STATE = "serialized_state"
LEGAL_ACTIONS = "legal_actions"
INFO_STATE = "info_state"


class OpenSpielProxy(worlds.Game):
    """Proxy to an OpenSpiel game instance."""

    def __init__(self, game: Union[str, pyspiel.Game], include_full_state: bool = False, **kwargs):
        """Initializes an `OpenSpielProxy`.

        Args:
            game: PySpiel game or a string representation of a game.
            include_full_state: Include the serialized state in the observation.
            **kwargs: Additional settings passed to the OpenSpiel game.
        """
        if isinstance(game, pyspiel.Game):
            logging.info("Proxy built using game instance: %s", game.get_type().short_name)
            self._game = game
        elif kwargs:
            game_settings = {key: pyspiel.GameParameter(val) for (key, val) in kwargs.items()}
            logging.info("Proxy built using game settings: %s", game_settings)
            self._game = pyspiel.load_game(game, game_settings)
        else:
            logging.info("Proxy built using game string: %s", game)
            self._game = pyspiel.load_game(game)
        self._game = rl_environment.Environment(self._game, include_full_state=include_full_state)

        self._num_players = self._game.num_players

    def reset(self) -> worlds.PlayerIDToTimestep:
        """Reset the game to it's initial state."""
        timesteps = self._game.reset()
        timesteps = self._convert_openspiel_timestep(timesteps)
        return timesteps

    def step(self, actions: _types.PlayerIDToAction) -> worlds.PlayerIDToTimestep:
        """Updates the game according to eahc players' action."""
        actions = list([actions[player_id] for player_id in range(self._num_players)])
        timesteps = self._game.step(actions)
        timesteps = self._convert_openspiel_timestep(timesteps)
        return timesteps

    def reward_specs(self) -> worlds.PlayerIDToSpec:
        """Describes the reward returned by the environment."""
        return {id: worlds.ArraySpec(shape=(), dtype=np.float32, name="reward") for id in range(self._num_players)}

    def observation_specs(self) -> worlds.PlayerIDToSpec:
        """Describes the observations provided by the environment.

        NOTE: If serialized-state is provided in the observation it is not described by the observation spec.
        Since it is a string-type that cannot be handled by computational graphs, it is not described, and
        it is expected that the user requesting the serialized-state will properly account for it.
        """
        openspiel_spec = self._game.observation_spec()

        spec = {
            "info_state": worlds.ArraySpec(shape=openspiel_spec["info_state"], dtype=np.float64, name="info_state"),
            "legal_actions": worlds.ArraySpec(shape=openspiel_spec["legal_actions"], dtype=bool, name="legal_actions"),
        }
        return {id: spec for id in range(self._num_players)}

    def action_specs(self) -> worlds.PlayerIDToSpec:
        """Describes the actions that should be provided to `step`."""
        openspiel_spec = self._game.action_spec()
        spec = worlds.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=openspiel_spec["min"],
            maximum=openspiel_spec["max"],
            name="action",
        )
        return {id: spec for id in range(self._num_players)}

    def _convert_openspiel_timestep(self, timesteps) -> worlds.PlayerIDToTimestep:
        """Convert OpenSpiel TimeSteps to MARL's TimeStep."""
        # Start by caching meta-data that does not have player-specific components.
        current_player = timesteps.observations[_CURRENT_PLAYER]
        del timesteps.observations[_CURRENT_PLAYER]

        # Remove serialized state from observations as default, since JAX graphs cannot handle strings.
        serialized_state = None
        if SERIALIZED_STATE in timesteps.observations:
            serialized_state = timesteps.observations[SERIALIZED_STATE]
        del timesteps.observations[SERIALIZED_STATE]

        # Collect the player-specific components in independent observations.
        observations = {
            id: {k: np.asarray(v[id]) for k, v in timesteps.observations.items()} for id in range(self._num_players)
        }
        if serialized_state:
            for id in observations.keys():
                observations[id][SERIALIZED_STATE] = serialized_state

        # Legal actions is an observation field listing the action IDs that are legal
        # for the current information state. This creates a dynamic length sequence which is problematic.
        # Convert that into a mask of the possible actions.
        for id in range(self._num_players):
            observations[id][LEGAL_ACTIONS] = self._legal_actions_to_mask(id, observations[id][LEGAL_ACTIONS])

        # Parse rewards independent of observation, because they can be globally None, or per-player scalars.
        if timesteps.rewards:
            rewards = {id: np.asarray(timesteps.rewards[id], dtype=np.float32) for id in range(self._num_players)}
        else:
            rewards = np.zeros(self._num_players, dtype=np.float32)

        step_type = worlds.StepType(timesteps.step_type.value)
        converted_timesteps = {
            id: worlds.TimeStep(step_type=step_type, reward=rewards[id], observation=observations[id])
            for id in range(self._num_players)
        }

        if current_player in [TurnId.SIMULTANEOUS, TurnId.TERMINAL]:
            return converted_timesteps

        elif current_player == TurnId.CHANCE:
            raise ValueError("Pass on chance node?")

        elif current_player in [TurnId.DEFAULT_PLAYER_ID, TurnId.MEAN_FIELD, TurnId.INVALID]:
            raise ValueError(f"Build timesteps for {current_player} is undefined.")

        else:
            # It's an individual player's turn.
            return {current_player: converted_timesteps[current_player]}

    def _legal_actions_to_mask(self, player_id: _types.PlayerID, legal_actions: _types.Array) -> _types.Array:
        """Converts a list of legal actions into a boolean mask for the action's legality."""
        mask = np.zeros(self.action_specs()[player_id].maximum + 1, dtype=bool)
        # Legal actions is empty at the end of the episode.
        if len(legal_actions):
            mask[legal_actions] = True
        return mask
