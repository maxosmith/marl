import jax
from absl.testing import absltest, parameterized
from marl_experiments.gathering import experiment

from marl import games
from marl.utils import spec_utils


class ExperimentTest(parameterized.TestCase):
    """Test cases for the IMPALA basic experiment."""

    def test_policy_graph(self):
        game = games.RockPaperScissors(num_stages=10)
        timestep = game.reset()[0]
        # https://github.com/deepmind/acme/blob/b0bcd57400e10de1ada280be0cc3614c33683773/acme/agents/jax/impala/networks.py
        # https://github.com/deepmind/acme/blob/b0bcd57400e10de1ada280be0cc3614c33683773/acme/tf/networks/atari.py#L148
        policy, _, initial_state, _ = experiment.build_computational_graphs(spec_utils.make_game_specs(game)[0])
        params = policy.init(jax.random.PRNGKey(42))
        state = initial_state.apply(None, jax.random.PRNGKey(42), None)
        action = policy.apply(params, jax.random.PRNGKey(42), timestep, state)


if __name__ == "__main__":
    absltest.main()
