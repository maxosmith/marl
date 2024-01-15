"""Tests for eval_arena."""
import numpy as np
from absl.testing import absltest, parameterized
from psro.test_utils import ScriptGame

from marl import bots, worlds
from marl.games import openspiel_proxy
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
        game_kwargs={"trajectory": timesteps, "expected_actions": actions},
        num_episodes=2,
    )
    arena = eval_arena.EvalArena(
        players=players,
        scenarios=scenario,
        counter=None,
        step_key=None,
        logger=logger,
    )

    results = arena.run_evaluation()
    tree_utils.assert_equals(expected_results, results)

  def test_leduc_poker(self):
    """Regression test of a real sequential move game."""
    players = {
        0: bots.RandomActionBot(num_actions=3),
        1: bots.RandomActionBot(num_actions=3),
    }
    scenario = eval_arena.EvaluationScenario(
        game_ctor=openspiel_proxy.OpenSpielProxy,
        game_kwargs={"game": "leduc_poker"},
        num_episodes=2,
    )
    arena = eval_arena.EvalArena(
        players=players,
        scenarios=scenario,
        counter=None,
        step_key=None,
        logger=None,
    )
    arena.run_evaluation()

  def test_sequential_move_return(self):
    """Test a sequential move game."""
    players = {
        0: bots.RandomActionBot(num_actions=3),
        1: bots.RandomActionBot(num_actions=3),
    }

    scenario = eval_arena.EvaluationScenario(
        game_ctor=ScriptGame,
        game_kwargs={
            "timesteps": [
                {0: worlds.TimeStep(step_type=worlds.StepType.FIRST, reward=1, observation=np.array([1]))},
                {1: worlds.TimeStep(step_type=worlds.StepType.FIRST, reward=2, observation=np.array([2]))},
                {0: worlds.TimeStep(step_type=worlds.StepType.MID, reward=3, observation=np.array([3]))},
                {1: worlds.TimeStep(step_type=worlds.StepType.MID, reward=4, observation=np.array([4]))},
                {0: worlds.TimeStep(step_type=worlds.StepType.LAST, reward=5, observation=np.array([5]))},
                {1: worlds.TimeStep(step_type=worlds.StepType.LAST, reward=6, observation=np.array([6]))},
            ],
        },
        num_episodes=2,
    )
    arena = eval_arena.EvalArena(
        players=players,
        scenarios=scenario,
        counter=None,
        step_key=None,
        logger=None,
    )
    results = arena.run_evaluation()
    tree_utils.assert_equals(
        [
            eval_arena.EpisodeResult(
                episode_length=5,
                episode_return={0: 9.0, 1: 12.0},
            )
        ],
        results,
    )


if __name__ == "__main__":
  absltest.main()
