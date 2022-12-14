"""Test suite for `marl.services.empirical_game.normal_form`."""
import numpy as np
from absl.testing import absltest, parameterized

from marl.services.empirical_games import normal_form


class NormalFormTest(parameterized.TestCase):
    """Test cases for `marl.services.empirical_game.normal_form`."""

    def test_two_players(self):
        """Basic API tests for a 2-Player ENFG."""
        enfg = normal_form.EmpiricalNFG(num_agents=2)
        enfg.add_payoffs(profile={0: 0, 1: 0}, payoffs={0: [0, 0], 1: [0, 1]})
        np.testing.assert_array_equal(enfg.game_matrix(), np.array([[[0.0, 0.5]]]))

        enfg.add_payoffs(profile={0: 1, 1: 0}, payoffs={0: [1, 2], 1: [3, 4, 5]})
        np.testing.assert_array_equal(enfg.game_matrix(), np.array([[[0.0, 0.5]], [[1.5, 4.0]]]))


if __name__ == "__main__":
    absltest.main()
