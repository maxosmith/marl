"""Test suite for `marl.games.openspiel_proxy`."""
import numpy as np
import pyspiel
from absl.testing import absltest, parameterized

from marl import worlds
from marl.games import openspiel_proxy

_RPS_STRING = "repeated_game(stage_game=matrix_rps(),num_repetitions=2)"


class OpenSpielProxyRPSTest(parameterized.TestCase):
  """Test cases for the `OpenSpielProxy` for Repeated RPS."""

  @parameterized.named_parameters(
      dict(testcase_name="game_instance", game=pyspiel.load_game(_RPS_STRING)),
      dict(testcase_name="game_string", game=_RPS_STRING),
  )
  def test_init(self, **kwargs):
    """Tests initialization of a game."""
    game = openspiel_proxy.OpenSpielProxy(**kwargs)
    # TODO(maxsmith): Accessing private variables of a proxy is bad practice, refactor.
    game_parameters = game._game._game.get_parameters()  # pylint: disable=protected-access
    self.assertEqual(2, game_parameters["num_repetitions"])
    self.assertEqual("matrix_rps", game_parameters["stage_game"]["name"])

  def test_specs(self):
    """Tests spec definitions of a game."""
    game = openspiel_proxy.OpenSpielProxy(_RPS_STRING)

    # Reward.
    reward_specs = game.reward_specs()
    self.assertLen(reward_specs, 2)
    for player_id in range(2):
      self.assertEqual((), reward_specs[player_id].shape)
      self.assertEqual(np.float32, reward_specs[player_id].dtype)

    # Observation.
    obs_specs = game.observation_specs()
    self.assertLen(obs_specs, 2)
    for player_id in range(2):
      self.assertEqual((6,), obs_specs[player_id]["info_state"].shape)
      self.assertEqual(np.float64, obs_specs[player_id]["info_state"].dtype)

    # Action.
    action_specs = game.action_specs()
    self.assertLen(action_specs, 2)
    for player_id in range(2):
      self.assertEqual((), action_specs[player_id].shape)
      self.assertEqual(np.int32, action_specs[player_id].dtype)

  def test_reset(self):
    """Tests resetting a new game instance."""
    game = openspiel_proxy.OpenSpielProxy(_RPS_STRING)
    timesteps = game.reset()
    self.assertLen(timesteps, 2)
    for player_id in range(2):
      np.testing.assert_array_equal(
          np.zeros(6, dtype=np.float32),
          timesteps[player_id].observation["info_state"],
      )
      self.assertEqual(0, timesteps[player_id].reward)
      self.assertEqual(worlds.StepType.FIRST, timesteps[player_id].step_type)

  def test_step(self):
    """Tests game transitions."""
    game = openspiel_proxy.OpenSpielProxy(_RPS_STRING)
    timesteps = game.reset()

    timesteps = game.step({0: 1, 1: 0})
    np.testing.assert_array_equal([0.0, 1.0, 0.0, 1.0, 0.0, 0.0], timesteps[0].observation["info_state"])
    np.testing.assert_array_equal([0.0, 1.0, 0.0, 1.0, 0.0, 0.0], timesteps[1].observation["info_state"])
    self.assertEqual(timesteps[0].reward, 1)
    self.assertEqual(timesteps[1].reward, -1)
    self.assertEqual(worlds.StepType.MID, timesteps[0].step_type)
    self.assertEqual(worlds.StepType.MID, timesteps[1].step_type)

    timesteps = game.step({0: 0, 1: 2})
    np.testing.assert_array_equal([1.0, 0.0, 0.0, 0.0, 0.0, 1.0], timesteps[0].observation["info_state"])
    np.testing.assert_array_equal([1.0, 0.0, 0.0, 0.0, 0.0, 1.0], timesteps[1].observation["info_state"])
    self.assertEqual(timesteps[0].reward, 1)
    self.assertEqual(timesteps[1].reward, -1)
    self.assertEqual(worlds.StepType.LAST, timesteps[0].step_type)
    self.assertEqual(worlds.StepType.LAST, timesteps[1].step_type)

  def test_sequential_game(self):
    """Test games with sequential actions."""
    game = openspiel_proxy.OpenSpielProxy("kuhn_poker")
    timesteps = game.reset()
    self.assertIn(0, timesteps)
    timesteps = game.step({0: 0})
    self.assertIn(1, timesteps)

  def test_kuhn_poker(self):
    """Tests step types of a non-simultaneous move game."""
    game = openspiel_proxy.OpenSpielProxy("kuhn_poker", {"seed": 42})

    timesteps = game.reset()
    self.assertLen(timesteps, 1)
    self.assertEqual(timesteps[0].step_type, worlds.StepType.FIRST)

    timesteps = game.step({0: 0})
    self.assertLen(timesteps, 1)
    self.assertEqual(timesteps[1].step_type, worlds.StepType.FIRST)

    timesteps = game.step({1: 0})
    self.assertLen(timesteps, 2)
    for timestep in timesteps.values():
      self.assertEqual(timestep.step_type, worlds.StepType.LAST)

  def test_leduc_poker(self):
    """Tests step types of a non-simultaneous move game."""
    game = openspiel_proxy.OpenSpielProxy("leduc_poker", {"seed": 42})

    timesteps = game.reset()
    self.assertLen(timesteps, 1)
    self.assertEqual(timesteps[0].step_type, worlds.StepType.FIRST)

    timesteps = game.step({0: 1})
    self.assertLen(timesteps, 1)
    self.assertEqual(timesteps[1].step_type, worlds.StepType.FIRST)

    timesteps = game.step({1: 1})
    self.assertLen(timesteps, 1)
    self.assertEqual(timesteps[0].step_type, worlds.StepType.MID)

    timesteps = game.step({0: 1})
    self.assertLen(timesteps, 1)
    self.assertEqual(timesteps[1].step_type, worlds.StepType.MID)

    timesteps = game.step({1: 0})
    self.assertLen(timesteps, 2)
    for timestep in timesteps.values():
      self.assertEqual(timestep.step_type, worlds.StepType.LAST)


if __name__ == "__main__":
  absltest.main()
