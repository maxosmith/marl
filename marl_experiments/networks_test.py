import haiku as hk
import jax
import numpy as np
from absl.testing import absltest, parameterized

from marl import _types, worlds
from marl.rl.agents.impala import example_networks, impala


class GraphsTest(parameterized.TestCase):
    """Test cases for the IMPALA computational graph."""

    @hk.testing.transform_and_run
    def test_step(self):
        impala = impala.IMPALA(
            timestep_encoder=example_networks.TimestepEncoder(),
            memory_core=example_networks.MemoryCore(),
            policy_head=example_networks.PolicyHead(num_actions=3),
            value_head=example_networks.ValueHead(),
        )
        timestep = worlds.TimeStep(
            step_type=worlds.StepType.FIRST,
            reward=0.0,
            observation=np.array([0, 0, 0, 0, 0, 0], dtype=float),
        )
        impala(timestep, impala.initial_state(None))


class NetworksTest(parameterized.TestCase):
    """Test cases for the IMPALA basic networks."""

    @hk.testing.transform_and_run
    def test_timestep_encoder(self):
        encoder = example_networks.TimestepEncoder()
        result = encoder(
            worlds.TimeStep(
                step_type=worlds.StepType.FIRST,
                reward=0.0,
                observation=np.array([0, 0, 0, 0, 0, 0], dtype=float),
            )
        )

    @hk.testing.transform_and_run
    def test_memory_core(self):
        core = example_networks.MemoryCore()
        inputs = np.ones([3])
        state = core.initial_state(None)
        result = core(inputs, state)

    @hk.testing.transform_and_run
    def test_policy_head(self):
        head = example_networks.PolicyHead(3)
        result = head(np.ones([3]))

    @hk.testing.transform_and_run
    def test_value_head(self):
        head = example_networks.ValueHead()
        result = head(np.ones([3]))

    @hk.testing.transform_and_run
    def test_regression(self):
        encoder = example_networks.TimestepEncoder()
        core = example_networks.MemoryCore()
        policy = example_networks.PolicyHead(3)
        value = example_networks.ValueHead()

        timestep = worlds.TimeStep(
            step_type=worlds.StepType.FIRST,
            reward=0.0,
            observation=np.array([0, 0, 0, 0, 0, 0], dtype=float),
        )

        h = encoder(timestep)
        h, _ = core(h, core.initial_state(None))
        _ = policy(h)
        _ = value(h)


if __name__ == "__main__":
    absltest.main()
