"""Tests for train_arena.

TODO:
  - Verify flow of data into and out of the game.
  - Verify that agents are responding correctly to timesteps.
"""
from absl.testing import absltest, parameterized

from marl import bots
from marl.games import openspiel_proxy
from marl.services.arenas import test_utils, train_arena

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
    expected_adds += [(trajectory[-1][learner_id], 0, ())]  # Zero is dummy action.

    game = test_utils.MockGame(trajectory=timesteps, expected_actions=actions)
    players = {}
    for player_i in range(2):
      players[player_i] = test_utils.MockActionSequenceBot(
          sequence=[joint[player_i] for joint in actions],
          expected_timesteps=[joint[player_i] for joint in timesteps],
      )
    adder = test_utils.MockAdder(expected_adds)
    logger = test_utils.MockLogger(expected_logs)

    arena = train_arena.TrainArena(
        game=game,
        learner_id=learner_id,
        players=players,
        adder=adder,
        logger=logger,
    )
    arena.run(num_episodes=1)

  @parameterized.parameters((0,), (1,))
  def test_full_trajectory_added(self, learner_id: int):
    """Tests if a full trajectory is added to the replay buffer."""
    players = {
        0: bots.RandomActionBot(num_actions=3),
        1: bots.RandomActionBot(num_actions=3),
    }
    game = openspiel_proxy.OpenSpielProxy(game="leduc_poker")
    adder = test_utils.MockStepTypeAdder()

    arena = train_arena.TrainArena(
        game=game,
        learner_id=learner_id,
        players=players,
        adder=adder,
        logger=test_utils.DummyLogger(),
    )
    arena.run(num_episodes=1)
    adder.verify()


if __name__ == "__main__":
  absltest.main()
