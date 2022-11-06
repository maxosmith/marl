"""Test suite for `marl_experiments.roshambo.roshambo_bot`."""
from absl.testing import absltest, parameterized

from marl.games import openspiel_proxy
from marl_experiments.roshambo import roshambo_bot

_NUM_THROWS = 2
_RPS_STRING = f"repeated_game(stage_game=matrix_rps(),num_repetitions={_NUM_THROWS})"


class RoshamboBotTest(parameterized.TestCase):
    """Test suite for RoshamboBot."""

    def test_basic_api(self):
        """Tests the basic API."""
        bot0 = roshambo_bot.RoshamboBot(name="rotatebot", num_throws=_NUM_THROWS)
        bot1 = roshambo_bot.RoshamboBot(name="rotatebot", num_throws=_NUM_THROWS)

        game = openspiel_proxy.OpenSpielProxy(_RPS_STRING, include_full_state=True)

        timesteps = game.reset()
        bot0.episode_reset(0, timesteps[0])
        bot1.episode_reset(1, timesteps[1])

        for _ in range(2):
            actions = {0: bot0.step(timesteps[0])[0], 1: bot1.step(timesteps[1])[0]}
            print(actions)
            timesteps = game.step(actions)


if __name__ == "__main__":
    absltest.main()
