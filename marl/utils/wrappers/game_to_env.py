import dataclasses

from marl import _types, worlds


@dataclasses.dataclass
class GameToEnv(worlds.Environment):
    """Abstract base class for Python RL environments.

    Observations and valid actions are described with `Array` specs, defined in
    the `specs` module.
    """

    game: worlds.Game
    player_id: _types.PlayerID

    def reset(self) -> worlds.TimeStep:
        """Starts a new sequence and returns the first `TimeStep` of this sequence."""
        return self.game.reset()[self.player_id]

    def step(self, action: _types.Action) -> worlds.TimeStep:
        """Updates the environment according to the action and returns a `TimeStep`."""
        return self.game.step({self.player_id: action})[self.player_id]

    def reward_spec(self) -> worlds.TreeSpec:
        """Describes the reward returned by the environment."""
        self.game.action_specs()[self.player_id]

    def observation_spec(self) -> worlds.TreeSpec:
        """Defines the observations provided by the environment."""
        return self.game.observation_specs()[self.player_id]

    def action_spec(self) -> worlds.TreeSpec:
        """Defines the actions that should be provided to `step`."""
        return self.game.action_specs()[self.player_id]
