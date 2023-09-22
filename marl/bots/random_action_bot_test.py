"""Test for `RandomActionBot`."""
import chex
from absl.testing import absltest, parameterized

from marl import worlds
from marl.bots import random_action_bot

_DUMMY_TIMESTEP = worlds.TimeStep(step_type=worlds.StepType.MID, reward=0, observation=0)


class RandomActionBotTest(parameterized.TestCase):
  """Test suite for `RandomActionBot`."""

  @parameterized.parameters(1, 3, 5, 10)
  def test_step(self, num_actions):
    """Tests `step` method."""
    bot = random_action_bot.RandomActionBot(num_actions=num_actions, seed=42)

    state = bot.episode_reset(_DUMMY_TIMESTEP)
    for _ in range(100):
      state, action = bot.step(state, _DUMMY_TIMESTEP)
      self.assertGreaterEqual(action, 0)
      self.assertLess(action, num_actions)

  @parameterized.parameters(chex.params_product(((1,), (3,), (5,), (10,)), ((0,), (1,), (42,))))
  def test_episode_reset(self, num_actions, seed):
    """Tests `episode_reset` method."""
    bot = random_action_bot.RandomActionBot(num_actions=num_actions, seed=seed)
    state = bot.episode_reset(_DUMMY_TIMESTEP)
    self.assertEqual(state, ())


if __name__ == "__main__":
  absltest.main()
