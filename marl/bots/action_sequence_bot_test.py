"""Tests for `ActionSequenceBot`."""
from typing import Sequence

from absl.testing import absltest, parameterized

from marl import types
from marl.bots import action_sequence_bot
from marl.utils import tree_utils

_TEST_CASES = (
    dict(
        action_sequence=(0,),
        expected_sequence=(0,),
    ),
    dict(
        action_sequence=(0,),
        expected_sequence=(0, 0, 0, 0),
    ),
    dict(
        action_sequence=(1, 2),
        expected_sequence=(1, 2, 1, 2, 1),
    ),
    dict(
        action_sequence=("rock", "paper", "scissors"),
        expected_sequence=("rock", "paper", "scissors", "rock", "paper"),
    ),
)


class ActionSequenceBotTest(parameterized.TestCase):
    """Test suite for `ActionSequenceBot`."""

    @parameterized.parameters(*_TEST_CASES)
    def test_step(
        self,
        action_sequence: Sequence[types.Action],
        expected_sequence: Sequence[types.Action],
    ):
        """Test the bot's `step` method."""
        bot = action_sequence_bot.ActionSequenceBot(sequence=action_sequence)
        state = bot.episode_reset(None)
        for expected_action in expected_sequence:
            taken_action, state = bot.step(state, None)
            tree_utils.assert_equals(taken_action, expected_action)

    @parameterized.parameters(*_TEST_CASES)
    def test_reset_sequence(
        self,
        action_sequence: Sequence[types.Action],
        expected_sequence: Sequence[types.Action],
    ):
        """Test the bot resets its sequence with the episode."""
        bot = action_sequence_bot.ActionSequenceBot(sequence=action_sequence)
        for _ in expected_sequence:
            state = bot.episode_reset(None)
            taken_action, state = bot.step(state, None)
            tree_utils.assert_equals(taken_action, expected_sequence[0])

    @parameterized.parameters(*_TEST_CASES)
    def test_episode_reset(
        self,
        action_sequence: Sequence[types.Action],
        expected_sequence: Sequence[types.Action],
    ):
        """Test the bot's `episode_reset` method."""
        del expected_sequence  # Used by `test_step`.
        bot = action_sequence_bot.ActionSequenceBot(sequence=action_sequence)
        initial_state = bot.episode_reset(None)
        tree_utils.assert_equals(initial_state, ())


if __name__ == "__main__":
    absltest.main()
