import functools
from typing import NamedTuple

import chex
import haiku as hk
import jax
import numpy as np
import rlax
import tree
from absl.testing import absltest, parameterized

from marl import _types, games
from marl.rl.agents.impala import example_networks, graphs


class ImpalaLossTest(parameterized.TestCase):
    """Test suite for `impala_loss`."""

    def setUp(self):
        super().setUp()

        self._behavior_policy_logits = np.array(
            [[[8.9, 0.7], [5.0, 1.0], [0.6, 0.1], [-0.9, -0.1]], [[0.3, -5.0], [1.0, -8.0], [0.3, 1.7], [4.7, 3.3]]],
            dtype=np.float32,
        )
        self._target_policy_logits = np.array(
            [[[0.4, 0.5], [9.2, 8.8], [0.7, 4.4], [7.9, 1.4]], [[1.0, 0.9], [1.0, -1.0], [-4.3, 8.7], [0.8, 0.3]]],
            dtype=np.float32,
        )
        self._actions = np.array([[0, 1, 0, 0], [1, 0, 0, 1]], dtype=np.int32)
        self._rho_tm1 = rlax.categorical_importance_sampling_ratios(
            self._target_policy_logits, self._behavior_policy_logits, self._actions
        )
        self._rewards = np.array([[-1.3, -1.3, 2.3, 42.0], [1.3, 5.3, -3.3, -5.0]], dtype=np.float32)
        self._values = np.array([[2.1, 1.1, -3.1, 0.0], [3.1, 0.1, -1.1, 7.4]], dtype=np.float32)
        self._bootstrap_value = np.array([8.4, -1.2], dtype=np.float32)

        self._expected_td = np.array([638.66376, 50.32439], dtype=np.float32)
        self._expected_pg = np.array([0.749191, -2.843706], dtype=np.float32)
        self._expected_entropy = np.array([-0.372598, -0.430028], dtype=np.float32)

    @chex.all_variants()
    @parameterized.parameters(
        [
            dict(baseline_cost=0.5, entropy_cost=0.02),
            dict(baseline_cost=0.6, entropy_cost=0.02),
            dict(baseline_cost=0.5, entropy_cost=0.04),
        ]
    )
    def test_impala_loss(self, baseline_cost: float, entropy_cost: float):
        """Basic API test."""
        impala_loss = self.variant(
            jax.vmap(
                functools.partial(
                    graphs.impala_loss, discount=0.99, baseline_cost=baseline_cost, entropy_cost=entropy_cost
                )
            )
        )
        mean_loss, metrics = impala_loss(
            logits=self._target_policy_logits,
            behaviour_logits=self._behavior_policy_logits,
            actions=self._actions,
            values_tm1=self._values,
            values_t=np.concatenate([self._values[:, 1:], self._bootstrap_value[:, None]], axis=1),
            rewards=self._rewards,
            mask=np.ones_like(self._rewards, dtype=bool),
        )

        # TD Loss.
        np.testing.assert_allclose(self._expected_td, metrics["critic_loss"], rtol=1e-3)
        np.testing.assert_allclose(baseline_cost * self._expected_td, metrics["scaled_critic_loss"], rtol=1e-3)

        # Policy Gradient Loss.
        np.testing.assert_allclose(self._expected_pg, metrics["policy_loss"], rtol=1e-3)

        # Entropy Loss.
        np.testing.assert_allclose(self._expected_entropy, metrics["entropy_loss"], rtol=1e-3)
        np.testing.assert_allclose(entropy_cost * self._expected_entropy, metrics["scaled_entropy_loss"], rtol=1e-3)

        # Total Loss.
        loss = self._expected_pg + baseline_cost * self._expected_td + entropy_cost * self._expected_entropy
        np.testing.assert_allclose(loss, mean_loss, rtol=1e-3)


if __name__ == "__main__":
    absltest.main()
