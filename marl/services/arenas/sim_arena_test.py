"""Tests for sim_arena."""
from absl.testing import absltest, parameterized

from marl import bots, worlds
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


if __name__ == "__main__":
  absltest.main()
