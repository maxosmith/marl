""" Tests for egta.envs.Gathering. """
from __future__ import absolute_import, division, print_function

import numpy as np
from absl.testing import absltest, parameterized

from marl.games.gathering import gathering

_UP = gathering.GatheringActions.UP.value
_DOWN = gathering.GatheringActions.DOWN.value
_LEFT = gathering.GatheringActions.LEFT.value
_RIGHT = gathering.GatheringActions.RIGHT.value
_ROTATE_LEFT = gathering.GatheringActions.ROTATE_LEFT.value
_ROTATE_RIGHT = gathering.GatheringActions.ROTATE_RIGHT.value
_LASER = gathering.GatheringActions.LASER.value
_NOOP = gathering.GatheringActions.NOOP.value


def print_observation(timesteps):
    flat_observation = timesteps[0].observation.copy()
    flat_observation[..., 0] *= 1
    flat_observation[..., 1] *= 2
    flat_observation[..., 2] *= 3
    flat_observation[..., 3] *= 4
    flat_observation[..., 4] *= 0
    flat_observation = np.sum(flat_observation, axis=-1)
    print(flat_observation[:, ::-1].T)


class GatheringTest(parameterized.TestCase):
    """Test cases for the `Gathering` game."""

    # def test_simultaneous_beams(self):
    #     """."""
    #     env = gathering.Gathering(n_agents=2, map_name="default_small")
    #     timesteps = env.reset()

    #     print(env.observation_specs())

    #     print(env.render(mode="text"))

    #     _ = env.step({0: _RIGHT, 1: _RIGHT})
    #     print(env.render(mode="text"))

    #     _ = env.step({0: _ROTATE_RIGHT, 1: _NOOP})
    #     print(env.render(mode="text"))

    #     _ = env.step({0: _LASER, 1: _NOOP})
    #     print(env.render(mode="text"))

    #     _ = env.step({0: _NOOP, 1: _NOOP})
    #     print(env.render(mode="text"))

    # def test_rotate(self):
    #     """."""
    #     env = gathering.Gathering(n_agents=2, map_name="default_small")
    #     timesteps = env.reset()
    #     print(env.render(mode="text"))

    #     _ = env.step({0: _DOWN, 1: _NOOP})
    #     print(env.render(mode="text"))

    #     _ = env.step({0: _ROTATE_RIGHT, 1: _NOOP})
    #     print(env.render(mode="text"))

    #     _ = env.step({0: _UP, 1: _NOOP})
    #     print(env.render(mode="text"))

    #     _ = env.step({0: _ROTATE_RIGHT, 1: _NOOP})
    #     print(env.render(mode="text"))

    #     _ = env.step({0: _UP, 1: _NOOP})
    #     print(env.render(mode="text"))

    # def test_vertical(self):
    #     """."""
    #     env = gathering.Gathering(n_agents=2, map_name="default_small")
    #     timesteps = env.reset()
    #     print(env.render(mode="text"))

    #     _ = env.step({0: _DOWN, 1: _NOOP})
    #     print(env.render(mode="text"))

    #     _ = env.step({0: _UP, 1: _NOOP})
    #     print(env.render(mode="text"))

    # def test_horizontal(self):
    #     env = gathering.Gathering(n_agents=2, map_name="default_small")
    #     timesteps = env.reset()
    #     print(env.render(mode="text"))

    #     _ = env.step({0: _DOWN, 1: _NOOP})
    #     print(env.render(mode="text"))

    #     _ = env.step({0: _RIGHT, 1: _NOOP})
    #     print(env.render(mode="text"))

    #     _ = env.step({0: _LEFT, 1: _NOOP})
    #     print(env.render(mode="text"))

    def test_observation(self):
        env = gathering.Gathering(n_agents=2, map_name="default_small", global_observation=False)
        timesteps = env.reset()
        print(env.render(mode="text"))
        print_observation(timesteps)

        timesteps = env.step({0: _DOWN, 1: _NOOP})
        print(env.render(mode="text"))
        print_observation(timesteps)

        timesteps = env.step({0: _ROTATE_RIGHT, 1: _NOOP})
        print(env.render(mode="text"))
        print_observation(timesteps)

        timesteps = env.step({0: _RIGHT, 1: _NOOP})
        print(env.render(mode="text"))
        print_observation(timesteps)

        timesteps = env.step({0: _LEFT, 1: _NOOP})
        print(env.render(mode="text"))
        print_observation(timesteps)

        timesteps = env.step({0: _ROTATE_LEFT, 1: _NOOP})
        print(env.render(mode="text"))
        print_observation(timesteps)


if __name__ == "__main__":
    absltest.main()
