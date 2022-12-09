"""Proxy that allows interacting with a world-model like a game."""
from typing import Optional

import haiku as hk
import jax

from marl import _types, services, worlds


class WorldModelGameProxy(worlds.Game):
    """Wrapper that allows interacting with a world-model like a game."""

    def __init__(
        self,
        transition: hk.Transformed,
        init_state: hk.Transformed,
        random_key: jax.random.KeyArray,
        game: worlds.Game,
        variable_source: Optional[services.VariableClient] = None,
        params: Optional[_types.Tree] = None,
    ):
        """Initialize an instance of a `WorldModelGameWrapper`."""
        if (variable_source is None) and (params is None):
            raise ValueError("Must specify either `variable_source` or `params`.")

        # World model graphs and parameters.
        self._transition = jax.jit(transition.apply)
        self._init_state = jax.jit(init_state.apply)
        self._variable_source = variable_source
        self._params = params
        self._random_key = random_key

        # Game used for specifications and initialization episodes.
        self._game = game

        # Memory/recurrent information.
        self._prev_timesteps = None
        self._memory = None

    def reset(self) -> worlds.PlayerIDToTimestep:
        """Starts a new sequence and returns the first `TimeStep` of this sequence.

        NOTE: We're currently not assuming a stochastic environment model, so we do
        not train it to sample initial states. Instead we use the true game for an
        initial state.
        """
        if self._variable_source:
            # Update model parameters at the end of each episode.
            self._variable_source.update(wait=True)
        self._memory = self._init_state(None, None, None)
        self._prev_timesteps = self._game.reset()
        return self._prev_timesteps

    def step(self, actions: _types.PlayerIDToAction) -> worlds.PlayerIDToTimestep:
        """Updates the environment according to the action and returns a `TimeStep`."""
        outputs = self._transition(
            self.params,
            self._random_key,
            world_state=self._prev_timesteps,
            actions=actions,
            memory=self._memory,
        )
        self._prev_timesteps, self._memory = outputs

        # The model outputs the classes, but agents expect one-hot.
        for player_id, timestep in self._prev_timesteps.items():
            self._prev_timesteps[player_id] = timestep._replace(
                observation=jax.nn.one_hot(timestep.observation, 5).astype(int)
            )

        return self._prev_timesteps

    def reward_specs(self) -> worlds.PlayerIDToSpec:
        """A specification of the reward space for each player."""
        return self._game.reward_specs()

    def observation_specs(self) -> worlds.PlayerIDToSpec:
        """A specification of the observation space for each player."""
        return self._game.observation_specs()

    def action_specs(self) -> worlds.PlayerIDToSpec:
        """A specification of the action space for each player."""
        return self._game.action_specs()

    @property
    def params(self):
        """Get the parameters for this model."""
        if self._params is not None:
            return self._params
        else:
            return self._variable_source.params
