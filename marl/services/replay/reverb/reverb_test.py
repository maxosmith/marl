"""Test cases for reverb-based replay buffers on launchpad nodes.

Launchpad's `ReverbNode` returns a reverb `Client`. A `Client` exposes
two main methods:
    - `insert` for inserting data into the buffer.
    - `sample` for sampling data from the buffer.

"""
from typing import Any, Optional

import launchpad as lp
import numpy as np
import reverb
from absl.testing import absltest

from marl import games
from marl.services.replay.reverb import adders as reverb_adders
from marl.utils import spec_utils

_TABLE = "table"


def priority_tables_fn(signature: Optional[Any] = None):
    return [
        reverb.Table(
            name=_TABLE,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=100,
            rate_limiter=reverb.rate_limiters.MinSize(10),
            signature=None,
        )
    ]


class ReverbNodeTest(absltest.TestCase):
    """Tests related to Reverb replay buffers."""

    def test_insert(self):
        program = lp.Program("test")
        reverb_handle = program.add_node(lp.ReverbNode(priority_tables_fn=priority_tables_fn), label="reverb")
        lp.launch(
            program,
            launch_type=lp.LaunchType.TEST_MULTI_THREADING,
            test_case=self,
            serialize_py_nodes=False,
        )
        reverb_client = reverb_handle.dereference()
        reverb_client.insert([np.zeros((81, 81))], {_TABLE: 1})

    def test_sequence_adder(self):
        sequence_length = 3
        game = games.RockPaperScissors(num_stages=10)

        signature = signature = reverb_adders.SequenceAdder.signature(
            spec_utils.make_game_specs(game)[0],
            (),
            sequence_length=sequence_length,
        )

        program = lp.Program("test")
        reverb_handle = program.add_node(
            lp.ReverbNode(priority_tables_fn=lambda: priority_tables_fn(signature=signature)), label="reverb"
        )
        lp.launch(
            program,
            launch_type=lp.LaunchType.TEST_MULTI_THREADING,
            test_case=self,
            serialize_py_nodes=False,
        )
        reverb_client = reverb_handle.dereference()
        reverb_adder = reverb_adders.SequenceAdder(
            client=reverb_client,
            priority_fns={_TABLE: None},
            sequence_length=sequence_length,
            period=sequence_length,
        )

        game = games.RockPaperScissors(num_stages=10)
        timesteps = game.reset()
        reverb_adder.add(timestep=timesteps[0])
        timesteps = game.step({0: 1, 1: 1})
        print(timesteps[0])
        reverb_adder.add(action=0, timestep=timesteps[0], extras=())
        timesteps = game.step({0: 1, 1: 1})
        reverb_adder.add(action=0, timestep=timesteps[0], extras=())


if __name__ == "__main__":
    absltest.main()
