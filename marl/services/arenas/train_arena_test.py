"""Tests for train_arena.

TODO:
  - Verify flow of data into and out of the game.
  - Verify that agents are responding correctly to timesteps.
"""
from absl.testing import absltest, parameterized

from marl import bots, worlds
from marl.games import mdp
from marl.services.arenas import test_utils, train_arena
from marl.services.replays import noop_adder

_LENGTH = "episode_length"
_RETURN = "episode_return/player_{i}"

_TWO_STEP_LOGS = (
    {
        _LENGTH: 2,
        _RETURN.format(i=0): 1.7,
        _RETURN.format(i=1): 0.25,
    },
)


class TrainArenaTest(parameterized.TestCase):
  """Test suite for `TrainArena`."""

  @parameterized.parameters(
      dict(
          trajectory=test_utils.TWO_STEP_TRAJ,
          learner_id=0,
          expected_logs=_TWO_STEP_LOGS,
      ),
      dict(
          trajectory=test_utils.TWO_STEP_TRAJ,
          learner_id=1,
          expected_logs=_TWO_STEP_LOGS,
      ),
  )
  def test_arena(self, trajectory, learner_id, expected_logs):
    """Test running the train arena."""
    timesteps = trajectory[::2]
    actions = trajectory[1::2]
    expected_adds = [
        (ts[learner_id], exp_actions[learner_id], ()) for ts, exp_actions in zip(trajectory[::2], trajectory[1::2])
    ]
    # Current last timestep is logged with previous actions copied forward.
    expected_adds += [(trajectory[-1][learner_id], trajectory[-2][learner_id], ())]

    game = test_utils.MockGame(trajectory=timesteps, expected_actions=actions)
    players = {}
    for player_i in range(2):
      players[player_i] = test_utils.MockActionSequenceBot(
          sequence=[joint[player_i] for joint in actions],
          expected_timesteps=[joint[player_i] for joint in timesteps],
      )
    adder = test_utils.MockAdder(expected_adds)
    logger = test_utils.MockLogger(expected_logs)

    arena = train_arena.TrainArena(game=game, adder=adder, logger=logger)
    arena.run(learner_id=learner_id, players=players, num_episodes=1)


if __name__ == "__main__":
  absltest.main()
