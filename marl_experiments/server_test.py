import collections
import functools
from pickletools import pyset
from typing import DefaultDict, List

import launchpad as lp
import numpy as np
import reverb
from absl import app
from courier.python import client, py_server
from dm_env import specs
from launchpad import context

from marl import worlds
from marl.utils import mocks, spec_utils
from marl_experiments import counter


def main(_):
    """Build and run the IMPALA distributed training topology."""
    game = mocks.DiscreteSymmetricSpecGame(num_actions=2, num_players=2)
    player_id_to_spec = spec_utils.make_game_specs(game)
    players = {id: mocks.Agent(spec) for id, spec in player_id_to_spec.items()}

    server = py_server.Server()
    server.Bind("add", lambda a, b: a)
    server.Start()

    worker = client.Client(server.address)

    print(worker.add(1, 2))
    print(worker.add("a", "b"))

    z = counter.Counter(10)
    print(worker.add(z, z))

    bounded = worlds.BoundedArraySpec((), np.int32, 0, 0)
    discrete = worlds.DiscreteArraySpec(num_values=10)

    worker.add(game._player_to_mock_env[0].action_spec(), None)

    worker.add(game._player_to_mock_env[0]._spec, None)


if __name__ == "__main__":
    app.run(main)
