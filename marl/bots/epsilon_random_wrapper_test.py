"""Test for `EpsilonRandomWrapper`."""
import numpy as np
from absl.testing import absltest, parameterized

from marl import bots, worlds
from marl.bots import epsilon_random_wrapper

_DUMMY_TIMESTEP = worlds.TimeStep(step_type=worlds.StepType.MID, reward=0, observation=0)


class EpsilonRandomWrapperTest(parameterized.TestCase):
  """Test suite for `EpsilonRandomWrapper`."""

  def test_step(self):
    """Tests `step` method."""
    bot = epsilon_random_wrapper.EpsilonRandomWrapper(
        bots.ConstantActionBot(0),
        num_actions=2,
        epsilon=0.03,
    )
    state = bot.episode_reset(_DUMMY_TIMESTEP)
    # The logits during episode reset are just a placeholder.
    np.testing.assert_almost_equal(np.array([0.5, 0.5]), state.logits)
    state, _ = bot.step(state, _DUMMY_TIMESTEP)
    np.testing.assert_almost_equal(np.array([0.03 * 0.5 + 0.97, 0.03 * 0.5]), state.logits)


if __name__ == "__main__":
  absltest.main()
