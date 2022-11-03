from absl.testing import absltest, parameterized

from marl import bots
from marl import strategy as strategy_lib
from marl import worlds

_MOCK_SPEC = worlds.EnvironmentSpec(observation=None, action=worlds.ArraySpec((), dtype=int), reward=None)


class StrategyTest(parameterized.TestCase):
    """Test suite for `Strategy`."""

    def test_basic_api(self):
        """Tests the basic API of strategy."""
        strategy = strategy_lib.Strategy(
            policies=[bots.ConstantIntAction(0, _MOCK_SPEC), bots.ConstantIntAction(1, _MOCK_SPEC)],
            mixture=[1.0, 0.0],
        )
        strategy.episode_reset(None)
        action, *_ = strategy.step(None, None)
        self.assertEqual(0, action)
        strategy.set_policy(1)
        action, *_ = strategy.step(None, None)
        self.assertEqual(1, action)


if __name__ == "__main__":
    absltest.main()
