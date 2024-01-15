"""Tests for sim_arena."""
import numpy as np
from absl.testing import absltest, parameterized
from psro.test_utils import ScriptGame

from marl import bots, worlds
from marl.games import openspiel_proxy
from marl.services.arenas import sim_arena, test_utils
from marl.utils import tree_utils


class SimArenaTest(parameterized.TestCase):
  """Test suite for `SimArena`."""

  @parameterized.parameters(
      dict(
          trajectory=test_utils.TWO_STEP_TRAJ,
          num_episodes=1,
          expected_results=[
              sim_arena.EpisodeResult(
                  episode_length=2,
                  episode_return={0: 1.7, 1: 0.25},
              )
          ],
      ),
      dict(
          trajectory=test_utils.TWO_STEP_TRAJ,
          num_episodes=2,
          expected_results=[
              sim_arena.EpisodeResult(
                  episode_length=2,
                  episode_return={0: 1.7, 1: 0.25},
              ),
              sim_arena.EpisodeResult(
                  episode_length=2,
                  episode_return={0: 1.7, 1: 0.25},
              ),
          ],
      ),
  )
  def test_arena(self, trajectory, num_episodes, expected_results):
    """Test running a simulation arena."""
    timesteps = trajectory[::2]
    actions = trajectory[1::2]

    game = test_utils.MockGame(trajectory=timesteps, expected_actions=actions)
    players = {}
    for player_i in range(2):
      players[player_i] = test_utils.MockActionSequenceBot(
          sequence=[joint[player_i] for joint in actions],
          expected_timesteps=[joint[player_i] for joint in timesteps],
      )

    arena = sim_arena.SimArena(game)
    sim_results = arena.run(players, num_episodes=num_episodes)
    tree_utils.assert_equals(sim_results, expected_results)

  def test_leduc_poker(self):
    """Regression test of a real sequential move game."""
    players = {
        0: bots.RandomActionBot(num_actions=3),
        1: bots.RandomActionBot(num_actions=3),
    }
    arena = sim_arena.SimArena(openspiel_proxy.OpenSpielProxy("leduc_poker"))
    arena.run_episode(players)

  def test_sequential_move_return(self):
    """Test a sequential move game."""
    players = {
        0: bots.RandomActionBot(num_actions=3),
        1: bots.RandomActionBot(num_actions=3),
    }
    game = ScriptGame([
        {0: worlds.TimeStep(step_type=worlds.StepType.FIRST, reward=1, observation=np.array([1]))},
        {1: worlds.TimeStep(step_type=worlds.StepType.FIRST, reward=2, observation=np.array([2]))},
        {0: worlds.TimeStep(step_type=worlds.StepType.MID, reward=3, observation=np.array([3]))},
        {1: worlds.TimeStep(step_type=worlds.StepType.MID, reward=4, observation=np.array([4]))},
        {0: worlds.TimeStep(step_type=worlds.StepType.LAST, reward=5, observation=np.array([5]))},
        {1: worlds.TimeStep(step_type=worlds.StepType.LAST, reward=6, observation=np.array([6]))},
    ])
    arena = sim_arena.SimArena(game)
    results = arena.run_episode(players)
    tree_utils.assert_equals(
        sim_arena.EpisodeResult(
            episode_length=5,
            episode_return={0: 9.0, 1: 12.0},
        ),
        results,
    )


if __name__ == "__main__":
  absltest.main()
