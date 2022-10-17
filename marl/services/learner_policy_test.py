import haiku as hk
import jax
from absl.testing import absltest, parameterized

from marl import games, worlds
from marl.rl.agents.impala import example_networks
from marl.services import learner_policy
from marl.utils import mocks, spec_utils


def build_computational_graphs(env_spec: worlds.EnvironmentSpec):
    from marl.rl.agents.impala import graphs

    timestep = spec_utils.zeros_from_spec(env_spec)

    def _impala():
        impala = graphs.IMPALA(
            timestep_encoder=example_networks.TimestepEncoder(),
            memory_core=example_networks.MemoryCore(),
            policy_head=example_networks.PolicyHead(num_actions=3),
            value_head=example_networks.ValueHead(),
        )

        def init():
            return impala(timestep, impala.initial_state(None))

        return init, (impala.__call__, impala.loss, impala.initial_state, impala.state_spec)

    hk_graphs = hk.multi_transform(_impala)
    policy = hk.Transformed(init=hk_graphs.init, apply=hk_graphs.apply[0])
    loss = hk.Transformed(init=hk_graphs.init, apply=hk_graphs.apply[1])
    initial_state = hk.Transformed(init=hk_graphs.init, apply=hk_graphs.apply[2])
    state_spec = hk.without_apply_rng(hk_graphs).apply[3](None)  # No parameters.
    return policy, loss, initial_state, state_spec


class LearnerPolicyTest(parameterized.TestCase):
    """Test cases for `LearnerPolicy`."""

    def test_learner_policy(self):
        random_key = jax.random.PRNGKey(42)
        game = games.RockPaperScissors(num_stages=10)

        policy_graph, _, initial_state_graph, state_spec = build_computational_graphs(
            spec_utils.make_game_specs(game)[0]
        )
        learner_update_client = mocks.VariableSource(
            variables={"policy": policy_graph.init(jax.random.PRNGKey(42))}, use_default_key=False
        )

        policy = learner_policy.LearnerPolicy(
            policy_fn=policy_graph,
            initial_state_fn=initial_state_graph,
            reverb_adder=None,
            variable_source=learner_update_client,
            variable_client_key="policy",
            variable_update_period=100,
            random_key=random_key,
            backend="cpu",
        )

        timestep = game.reset()[0]
        state = policy.episode_reset(timestep)
        action, new_state = policy.step(timestep, state)


if __name__ == "__main__":
    absltest.main()
