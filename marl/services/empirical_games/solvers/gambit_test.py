"""Test suite for `marl.services.empirical_games.solvers.gambit`. """
import numpy as np
from absl.testing import absltest, parameterized

from marl.services.empirical_games.solvers import gambit


class GambitTest(parameterized.TestCase):
    """Test suite for `marl.services.empirical_games.solvers.gambit`."""

    def test_basic_regression(self):
        """Basic regression for `GambitAnalyzer`."""
        payoff_matrix = np.array([[[10, -10], [0, 0]], [[0, 0], [0, 0]]])

        solver = gambit.Gambit(timeout=120)
        nash = solver(payoff_matrix)
        np.testing.assert_array_equal(nash[0], [1, 0])
        np.testing.assert_array_equal(nash[1], [0, 1])

    def test_mixed_equilibrium(self):
        """Regression test with only mixed-strategy equilibriums."""
        po_row = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1]])
        po_col = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0]])
        payoff = np.stack([po_row, po_col], axis=2)

        solver = gambit.Gambit(timeout=120)
        nash = solver(payoff)
        np.testing.assert_array_equal(nash[0], [0.5, 0.5, 0, 0])
        np.testing.assert_array_equal(nash[1], [0.5, 0.5, 0, 0])

    def test_policy_permutation(self):
        """Ensure that Gambit correctly maintains policy ordering by testing swaps."""
        # Regular.
        payoff_matrix = np.array(
            [
                [[0.91, 2.55], [0.0, 568.18], [0.0, 562.55]],
                [[501.09, 0.45], [230.55, 302.27], [162.73, 379.27]],
                [[521.45, 0.0], [268.73, 272.91], [184.82, 376.91]],
            ]
        )

        solver = gambit.Gambit(timeout=120)
        nash = solver(payoff_matrix)
        np.testing.assert_array_equal(nash[0], [0, 0, 1])
        np.testing.assert_array_equal(nash[1], [0, 0, 1])

        # Swap rows 1 and 2.
        payoff_matrix = np.array(
            [
                [[0.91, 2.55], [0.0, 568.18], [0.0, 562.55]],
                [[521.45, 0.0], [268.73, 272.91], [184.82, 376.91]],
                [[501.09, 0.45], [230.55, 302.27], [162.73, 379.27]],
            ]
        )

        solver = gambit.Gambit(timeout=120)
        nash = solver(payoff_matrix)
        np.testing.assert_array_equal(nash[0], [0, 1, 0])
        np.testing.assert_array_equal(nash[1], [0, 0, 1])

        # Swap cols 1 and 2.
        payoff_matrix = np.array(
            [
                [[0.91, 2.55], [0.0, 562.55], [0.0, 568.18]],
                [[501.09, 0.45], [162.73, 379.27], [230.55, 302.27]],
                [[521.45, 0.0], [184.82, 376.91], [268.73, 272.91]],
            ]
        )

        solver = gambit.Gambit(timeout=120)
        nash = solver(payoff_matrix)
        np.testing.assert_array_equal(nash[0], [0, 0, 1])
        np.testing.assert_array_equal(nash[1], [0, 1, 0])

        # Swap rows 1 and 2 and cols 1 and 2.
        payoff_matrix = np.array(
            [
                [[0.91, 2.55], [0.0, 562.55], [0.0, 568.18]],
                [[521.45, 0.0], [184.82, 376.91], [268.73, 272.91]],
                [[501.09, 0.45], [162.73, 379.27], [230.55, 302.27]],
            ]
        )

        solver = gambit.Gambit(timeout=120)
        nash = solver(payoff_matrix)
        np.testing.assert_array_equal(nash[0], [0, 1, 0])
        np.testing.assert_array_equal(nash[1], [0, 1, 0])

    def test_uneven_action_spaces(self):
        """Test a non-square matrix."""
        payoff_matrix = np.array(
            [
                [[0.91, 2.55], [0.0, 568.18], [0.0, 562.55]],
                [[501.09, 0.45], [230.55, 302.27], [230.55, 302.27]],
            ]
        )

        solver = gambit.Gambit(timeout=120)
        nash = solver(payoff_matrix)
        np.testing.assert_array_equal(nash[0], [0, 1])
        np.testing.assert_array_equal(nash[1], [0, 1, 0])


if __name__ == "__main__":
    absltest.main()
