""" Tests for egta.envs.Gathering. """
from __future__ import absolute_import, division, print_function

from absl.testing import absltest, parameterized

from marl.games.gathering import gathering

_RIGHT = gathering.GatheringActions.RIGHT.value
_ROTATE_RIGHT = gathering.GatheringActions.ROTATE_LEFT.value
_LASER = gathering.GatheringActions.LASER.value
_NOOP = gathering.GatheringActions.NOOP.value


class GatheringTest(parameterized.TestCase):
    """Test cases for the `Gathering` game."""

    def test_simultaneous_beams(self):
        """."""
        env = gathering.Gathering(n_agents=2, map_name="default_small")
        timesteps = env.reset()

        print(env.observation_specs())

        print(env.render(mode="text"))

        _ = env.step({0: _RIGHT, 1: _RIGHT})
        print(env.render(mode="text"))

        _ = env.step({0: _ROTATE_RIGHT, 1: _NOOP})
        print(env.render(mode="text"))

        _ = env.step({0: _LASER, 1: _NOOP})
        print(env.render(mode="text"))

        _ = env.step({0: _NOOP, 1: _NOOP})
        print(env.render(mode="text"))


if __name__ == "__main__":
    absltest.main()
