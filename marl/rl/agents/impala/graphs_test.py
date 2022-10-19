from typing import NamedTuple

import haiku as hk
import numpy as np
import tree
from absl.testing import absltest, parameterized

from marl import _types, games
from marl.rl.agents.impala import example_networks, graphs


class MockStep(NamedTuple):
    action: _types.Tree
    observation: _types.Tree
    reward: _types.Tree
    extras: _types.Tree


_RPS_SAMPLE = MockStep(
    action=np.array([[0, 1, 2], [2, 1, 0]]),
    observation=np.array(
        [
            [[0, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0], [0, 1, 0, 1, 0, 0]],
            [[0, 0, 1, 0, 1, 0], [0, 1, 0, 0, 0, 1], [0, 0, 1, 1, 0, 0]],
        ]
    ),
    reward=np.array([[1, 1, 1], [1, 1, 1]]),
    extras=graphs.IMPALAState(
        recurrent_state=hk.LSTMState(
            cell=np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]),
            hidden=np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]),
        ),
        logits=np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]),
        prev_action=np.array([[-1, 0, 1], [-1, 2, 1]]),
    ),
)


class GraphsTest(parameterized.TestCase):
    @hk.testing.transform_and_run
    def test_loss(self):
        impala = graphs.IMPALA(
            timestep_encoder=example_networks.TimestepEncoder(),
            memory_core=example_networks.MemoryCore(),
            policy_head=example_networks.PolicyHead(num_actions=3),
            value_head=example_networks.ValueHead(),
            evaluation=True,
        )
        loss, metrics = impala.loss(_RPS_SAMPLE)
        print(metrics)


if __name__ == "__main__":
    absltest.main()
