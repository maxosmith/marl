"""Regression test for running a reverb replay buffer."""
import numpy as np
import reverb
from absl import app
from tqdm import trange

from marl.rl.replay.reverb import adders, dataset
from marl.utils import mocks, spec_utils


def main(_):
    env = mocks.DiscreteEnvironment(num_actions=3, obs_shape=(10,))
    env_spec = spec_utils.make_environment_spec(env)

    agent = mocks.Agent(env_spec)

    replay_table = reverb.Table(
        name="default",
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=1_000_000,
        rate_limiter=reverb.rate_limiters.MinSize(1),
        # signature=adders.NStepTransitionAdder.signature(env_spec),
        signature=adders.SequenceAdder.signature(env_spec, sequence_length=2),
    )
    replay_server = reverb.Server([replay_table], port=None)
    replay_client = reverb.Client(f"localhost:{replay_server.port}")
    # adder = adders.NStepTransitionAdder(priority_fns={"default": None}, client=replay_client, n_step=1, discount=1,)
    adder = adders.SequenceAdder(client=replay_client, sequence_length=2, period=2, priority_fns={"default": None})

    timestep = env.reset()
    adder.add(timestep=timestep)
    for steps in trange(10_000):
        action = agent.step(timestep)
        timestep = env.step(action)
        adder.add(timestep=timestep, action=action, extras=())

    ds = dataset.make_reverb_dataset(
        table="default",
        server_address=replay_client.server_address,
        batch_size=64,
        prefetch_size=4,
    )
    it = ds.as_numpy_iterator()

    for _ in trange(3):
        _ = next(it)


if __name__ == "__main__":
    app.run(main)
