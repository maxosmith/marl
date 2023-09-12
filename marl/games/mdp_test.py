"""Tests for MDP."""
from absl.testing import absltest, parameterized

from marl import worlds
from marl.games import mdp


class MDPTest(parameterized.TestCase):
  """Test suite for `MDP`."""

  def test_single_agent_mdp(self):
    """Basic compilation test for a single-agent MDP."""
    env = mdp.MDP(
        dynamics=[
            mdp.Transition(0, 0, 1, 0),
            mdp.Transition(0, 1, 2, 0),
            mdp.Transition(2, 0, 3, 1),
        ],
        terminal_states={3},
    )

    timestep: worlds.TimeStep = env.reset()
    self.assertEqual(timestep.step_type, worlds.StepType.FIRST)
    self.assertEqual(timestep.observation, 0)
    self.assertEqual(timestep.reward, 0)

    timestep: worlds.TimeStep = env.step(0)
    self.assertEqual(timestep.step_type, worlds.StepType.MID)
    self.assertEqual(timestep.observation, 1)
    self.assertEqual(timestep.reward, 0)

    timestep: worlds.TimeStep = env.reset()
    self.assertEqual(timestep.step_type, worlds.StepType.FIRST)
    self.assertEqual(timestep.observation, 0)
    self.assertEqual(timestep.reward, 0)

    timestep: worlds.TimeStep = env.step(1)
    self.assertEqual(timestep.step_type, worlds.StepType.MID)
    self.assertEqual(timestep.observation, 2)
    self.assertEqual(timestep.reward, 0)

    timestep: worlds.TimeStep = env.step(0)
    self.assertEqual(timestep.step_type, worlds.StepType.LAST)
    self.assertEqual(timestep.observation, 3)
    self.assertEqual(timestep.reward, 1)


if __name__ == "__main__":
  absltest.main()
