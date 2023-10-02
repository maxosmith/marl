"""Test for `jax_bot`."""
import jax
from absl.testing import absltest, parameterized

from marl.bots import jax_bot, test_utils


class JAXBotTest(parameterized.TestCase):
  """Test suite for `JAXBot`."""

  def test_step(self):
    """Tests step method."""
    bot = jax_bot.JAXBot(
        policy=test_utils.TestPolicyLinear(),
        params={"params": {"mean": 2}},
        rng_key=jax.random.PRNGKey(42),
    )
    state = bot.episode_reset(test_utils.FAKE_TIMESTEP)
    self.assertEqual(state, 1)
    action, state = bot.step(state, test_utils.FAKE_TIMESTEP)
    self.assertEqual(state, 2)
    self.assertEqual(action, 3)


if __name__ == "__main__":
  absltest.main()
