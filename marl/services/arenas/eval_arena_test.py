"""Tests for eval_arena."""
from absl.testing import absltest, parameterized

from marl.services.arenas import eval_arena, test_utils
from marl.utils import tree_utils


class EvalArenaTest(parameterized.TestCase):
  """Test suite for `EvalArena`."""

  @parameterized.parameters(
      dict(
          trajectory=test_utils.TWO_STEP_TRAJ,
          expected_results=[
              eval_arena.EpisodeResult(
                  episode_length=2,
                  episode_return={0: 1.7, 1: 0.25},
              )
          ],
      )
  )
  def test_single_scenario(self, trajectory, expected_results):
    """Test running the arena."""
    timesteps = trajectory[::2]
    actions = trajectory[1::2]

    players = {}
    for player_i in range(2):
      players[player_i] = test_utils.MockActionSequenceBot(
          sequence=[joint[player_i] for joint in actions],
          expected_timesteps=[joint[player_i] for joint in timesteps],
      )
    logger = test_utils.MockLogger([x.to_logdata() for x in expected_results])

    scenario = eval_arena.EvaluationScenario(
        game_ctor=test_utils.MockGame,
        game_kwargs=dict(trajectory=timesteps, expected_actions=actions),
        num_episodes=2,
    )
    arena = eval_arena.EvalArena(
        scenarios=scenario,
        logger=logger,
    )

    results = arena.run_evaluation(players)
    tree_utils.assert_equals(results, expected_results)


if __name__ == "__main__":
  absltest.main()
