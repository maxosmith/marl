"""Test suite for `marl.games.meltingpot_proxy`."""
import numpy as np
from absl.testing import absltest, parameterized
from meltingpot.python import substrate

from marl.games import meltingpot_proxy


_CLEAN_UP = "clean_up"


class MeltingPotProxyTest(parameterized.TestCase):
    """Test cases for `MeltingPotProxy`."""

    def test_clean_up(self):
        """Tests default Clean-Up substrate as a regression test."""
        config = substrate.get_config(_CLEAN_UP)
        config.num_players = 2
        config.lab2d_settings.numPlayers = 2
        config.lab2d_settings.simulation.gameObjects = config.lab2d_settings.simulation.gameObjects[:2]

        game = meltingpot_proxy.MeltingPotProxy(config)

        timesteps = game.reset()
        timesteps = game.step({0: 1, 1: 1})

        obs_specs = game.observation_specs()
        self.assertEqual(config.num_players, len(obs_specs))
        for player_id in range(config.num_players):
            self.assertIn(player_id, obs_specs)
            spec = obs_specs[player_id]
            np.testing.assert_array_equal((88, 88, 3), spec.shape)
            self.assertEqual(np.uint8, spec.dtype)

        reward_specs = game.reward_specs()
        self.assertEqual(config.num_players, len(reward_specs))
        for player_id in range(config.num_players):
            self.assertIn(player_id, reward_specs)
            spec = reward_specs[player_id]
            np.testing.assert_array_equal((), spec.shape)
            self.assertEqual(np.float64, spec.dtype)

        action_specs = game.action_specs()
        self.assertEqual(config.num_players, len(action_specs))
        for player_id in range(config.num_players):
            self.assertIn(player_id, action_specs)
            spec = action_specs[player_id]
            np.testing.assert_array_equal((), spec.shape)
            self.assertEqual(np.int32, spec.dtype)
            self.assertEqual(9, spec.num_values)


if __name__ == "__main__":
    absltest.main()
