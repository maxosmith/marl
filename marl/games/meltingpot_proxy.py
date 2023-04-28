"""Proxy to a MeltingPot game instance.

Available MeltingPot games can be found by running:
```
from meltingpot.python import substrate
print(substrate.AVAILABLE_SUBSTRATES)
```

Game configs are gotten by running `substrate.get_config(NAME)`.

Useful configs:
config.lab2d_settings.spriteSize --> 8
config.lab2d_settings.maxEpisodeLengthFrames --> 1_000
config.lab2d_settings.simulation.gameObjects[0]["components"]
    --> List of components, need "Avatar"
config.lab2d_settings.simulation.gameObjects[0]["components"][-6]["kwargs"]["view"]
"""
import enum
from marl import worlds, _types
from ml_collections import config_dict
import dm_env
from meltingpot.python import substrate
from typing import Union, Any
import numpy as np

_OBSERVATION_KEY = "RGB"
_GLOBAL_KEY = "WORLD.RGB"


class RenderModes(enum.Enum):
    """Render modes for `MeltingPotProxy`."""

    GLOBAL = "global"
    LOCAL = "local"


class MeltingPotProxy(worlds.Game):
    """Proxy to a MeltingPot game instance."""

    def __init__(self, game_config: config_dict.ConfigDict):
        """Initializes an `MeltingPotProxy`.

        Args:
            game_config: Config specifying a substrate (game) from MeltingPot.
                Template configs are gotten from `meltinpot.substrate.get_config`.
        """
        self._game_config = game_config
        self._game = substrate.build(game_config)
        self._prev_raw_timesteps = None

    def reset(self) -> worlds.PlayerIDToTimestep:
        """Reset the game to it's initial state."""
        timesteps = self._game.reset()
        self._prev_raw_timesteps = timesteps
        timesteps = self._convert_timesteps(timesteps)
        return timesteps

    def step(self, actions: _types.PlayerIDToAction) -> worlds.PlayerIDToTimestep:
        """Updates the game according to eahc players' action."""
        actions = list([actions[player_id] for player_id in range(self.num_players)])
        timesteps = self._game.step(actions)
        self._prev_raw_timesteps = timesteps
        timesteps = self._convert_timesteps(timesteps)
        return timesteps

    def reward_specs(self) -> worlds.PlayerIDToSpec:
        """Describes the reward returned by the environment."""
        return {
            player_id: worlds.ArraySpec(shape=(), dtype=np.float32, name="reward")
            for player_id in range(self.num_players)
        }

    def observation_specs(self) -> worlds.PlayerIDToSpec:
        """Describes the observations provided by the environment."""
        specs = {}
        for player_id, player_spec in enumerate(self._game.observation_spec()):
            specs[player_id] = worlds.ArraySpec(
                shape=player_spec[_OBSERVATION_KEY].shape,
                dtype=np.int32,
                name=player_spec[_OBSERVATION_KEY].name,
            )
        return specs

    def action_specs(self) -> worlds.PlayerIDToSpec:
        """Describes the actions that should be provided to `step`."""
        specs = self._game.action_spec()
        return {player_id: specs[player_id] for player_id in range(self.num_players)}

    @property
    def num_players(self):
        """Get the number of players in the game."""
        return self._game_config.num_players

    def render(self, mode: Union[RenderModes, str]) -> Any:
        """Get a view of the game state to render.

        Args:
            mode: Rendering mode specifying what game details to render.

        Returns:
            Game rendering depending on mode:
                * GLOBAL:
                * LOCAL:
        """
        if self._prev_raw_timesteps is None:
            raise ValueError("Cannot call render before `reset`.")

        if isinstance(mode, str):
            mode = RenderModes(mode)

        if mode == RenderModes.GLOBAL:
            return self._render_global()
        elif mode == RenderModes.LOCAL:
            return self._render_local()
        else:
            raise NotImplementedError(f"Rendering not defined for {mode=}.")

    def _render_global(self) -> Any:
        """Get a global state of the game."""
        return self._prev_raw_timesteps.observation[0][_GLOBAL_KEY]

    def _render_local(self) -> Any:
        """Get local states of the game."""
        observations = {}
        for player_id in range(self.num_players):
            observations[player_id] = self._prev_raw_timesteps.observation[player_id][_OBSERVATION_KEY]
        return observations

    def _convert_timesteps(self, timesteps: dm_env.TimeStep) -> worlds.PlayerIDToTimestep:
        """Convert MeltingPot timesteps to `marl` TimeSteps.

        Args:
            timesteps: MeltingPot raw timesteps.

        Returns:
            Timesteps that are `marl` compatible.
        """
        marl_timesteps = {}
        for player_id in range(self.num_players):
            marl_timesteps[player_id] = worlds.TimeStep(
                step_type=worlds.StepType(timesteps.step_type.value),
                observation=timesteps.observation[player_id][_OBSERVATION_KEY].astype(np.int32),
                reward=timesteps.reward[player_id].astype(np.float32),
            )
        return marl_timesteps
