import numpy as np
from absl.testing import absltest, parameterized

from marl import types, worlds
from marl.games import rock_paper_scissors

_ROCK = rock_paper_scissors.RockPaperScissorsActions.ROCK.value
_PAPER = rock_paper_scissors.RockPaperScissorsActions.PAPER.value
_SCISSORS = rock_paper_scissors.RockPaperScissorsActions.SCISSORS.value


class RockPaperScissorsTest(parameterized.TestCase):
  """Test cases for the `RockPaperScissors` game."""

  @parameterized.parameters(1, 2, 5)
  def test_init(self, num_stages: int):
    """Test init method."""
    game = rock_paper_scissors.RockPaperScissors(num_stages=num_stages)
    timesteps = game.reset()
    self.assertTrue(np.all([ts.first() for ts in timesteps.values()]))
    for _ in range(num_stages - 1):
      timesteps = game.step({0: 0, 1: 0})
      self.assertTrue(np.all([ts.mid() for ts in timesteps.values()]))
    timesteps = game.step({0: 0, 1: 0})
    self.assertTrue(np.all([ts.last() for ts in timesteps.values()]))

  @parameterized.parameters(
      dict(
          action0=_ROCK,
          action1=_ROCK,
          reward0=0,
          reward1=0,
          observation=[1, 0, 0, 1, 0, 0],
      ),
      dict(
          action0=_PAPER,
          action1=_PAPER,
          reward0=0,
          reward1=0,
          observation=[0, 1, 0, 0, 1, 0],
      ),
      dict(
          action0=_SCISSORS,
          action1=_SCISSORS,
          reward0=0,
          reward1=0,
          observation=[0, 0, 1, 0, 0, 1],
      ),
      dict(
          action0=_ROCK,
          action1=_PAPER,
          reward0=-1,
          reward1=1,
          observation=[1, 0, 0, 0, 1, 0],
      ),
      dict(
          action0=_ROCK,
          action1=_SCISSORS,
          reward0=1,
          reward1=-1,
          observation=[1, 0, 0, 0, 0, 1],
      ),
      dict(
          action0=_PAPER,
          action1=_ROCK,
          reward0=1,
          reward1=-1,
          observation=[0, 1, 0, 1, 0, 0],
      ),
      dict(
          action0=_PAPER,
          action1=_SCISSORS,
          reward0=-1,
          reward1=1,
          observation=[0, 1, 0, 0, 0, 1],
      ),
      dict(
          action0=_SCISSORS,
          action1=_ROCK,
          reward0=-1,
          reward1=1,
          observation=[0, 0, 1, 1, 0, 0],
      ),
      dict(
          action0=_SCISSORS,
          action1=_PAPER,
          reward0=1,
          reward1=-1,
          observation=[0, 0, 1, 0, 1, 0],
      ),
  )
  def test_step(
      self,
      action0: rock_paper_scissors.RockPaperScissorsActions,
      action1: rock_paper_scissors.RockPaperScissorsActions,
      reward0: float,
      reward1: float,
      observation: types.Array,
  ):
    """Test step method."""
    game = rock_paper_scissors.RockPaperScissors(num_stages=1)
    _ = game.reset()
    timesteps = game.step({0: action0, 1: action1})
    self.assertEqual(timesteps[0].reward, reward0)
    self.assertEqual(timesteps[1].reward, reward1)
    np.testing.assert_array_equal(timesteps[0].observation, observation)
    np.testing.assert_array_equal(timesteps[1].observation, observation)

  def test_reset(self):
    """Test RockPaperScissor's reset method."""
    game = rock_paper_scissors.RockPaperScissors(num_stages=1)
    timesteps = game.reset()
    self.assertLen(timesteps, 2)
    self.assertIn(0, timesteps)
    self.assertIn(1, timesteps)
    for timestep in timesteps.values():
      self.assertEqual(timestep.step_type, worlds.StepType.FIRST)
      self.assertEqual(timestep.reward, 0.0)
      np.testing.assert_array_equal(timestep.observation, np.zeros((6,), dtype=int))

  def test_specs(self):
    """Test RockPaperScissor's {reward|observation|action}_spec."""
    game = rock_paper_scissors.RockPaperScissors(num_stages=1)

    # Reward.
    reward_specs = game.reward_specs()
    self.assertIn(0, reward_specs)
    self.assertIn(1, reward_specs)
    for spec in reward_specs.values():
      self.assertEqual(spec.shape, ())
      self.assertEqual(spec.dtype, np.float32)

    # Observation.
    obs_specs = game.observation_specs()
    self.assertIn(0, obs_specs)
    self.assertIn(1, obs_specs)
    for spec in obs_specs.values():
      self.assertEqual(spec.shape, (6,))
      self.assertEqual(spec.dtype, np.int32)

    # Action.
    action_specs = game.action_specs()
    self.assertIn(0, action_specs)
    self.assertIn(1, action_specs)
    for spec in action_specs.values():
      self.assertEqual(spec.shape, ())
      self.assertEqual(spec.dtype, np.int32)


if __name__ == "__main__":
  absltest.main()
