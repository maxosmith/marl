"""Tests for `ConstantActionBot`."""
from absl.testing import absltest, parameterized

from marl.bots import constant_action_bot
from marl.utils import tree_utils

_TEST_CASES = ((0), (1.3), ("rock"), ("paper"))


class ConstantActionBotTest(parameterized.TestCase):
    """Test suite for `ConstantActionBot`."""

    @parameterized.parameters(*_TEST_CASES)
    def test_step(self, action):
        """Test the bot's `step` method."""
        bot = constant_action_bot.ConstantActionBot(action=action)
        state = bot.episode_reset(None)
        for _ in range(10):
            taken_action, state = bot.step(state, None)
            tree_utils.assert_equals(taken_action, action)

    @parameterized.parameters(*_TEST_CASES)
    def test_episode_reset(self, action):
        """Test the bot's `episode_reset` method."""
        bot = constant_action_bot.ConstantActionBot(action=action)
        initial_state = bot.episode_reset(None)
        tree_utils.assert_equals(initial_state, ())


if __name__ == "__main__":
    absltest.main()
