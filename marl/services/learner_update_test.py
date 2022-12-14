from typing import List, Sequence

import haiku as hk
import jax
import optax
import reverb
import tensorflow as tf
import tree
from absl.testing import absltest

from marl import games, worlds
from marl.agents.impala import example_networks
from marl.services import learner_update
from marl.services.replay.reverb import adders as reverb_adders
from marl.utils import distributed_utils, spec_utils

_TABLE = "table"


def build_computational_graphs(env_spec: worlds.EnvironmentSpec):
    from marl.agents.impala import impala

    timestep = spec_utils.zeros_from_spec(env_spec)

    def _impala():
        impala = impala.IMPALA(
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


def sequence_dataset_from_spec(
    env_spec: worlds.EnvironmentSpec,
    state_spec: worlds.TreeSpec,
    sequence_length: int,
    batch_size: int,
    devices: Sequence[jax.xla.Device],
) -> tf.data.Dataset:
    """Constructs fake dataset of Reverb sequence samples.

    Args:
        spec: Environment specification.
        sequence_length: Length of the sequences sampled from the dataset.

    Returns:
        tf.data.Dataset that produces the fake batches of sequence ReverbSample objects indefinitely.
    """
    signature = reverb_adders.SequenceAdder.signature(env_spec, state_spec, sequence_length=sequence_length)
    data = spec_utils.generate_from_tf_spec(signature)
    info = tree.map_structure(lambda tf_dtype: tf.ones([], tf_dtype.as_numpy_dtype), reverb.SampleInfo.tf_dtypes())
    sample = reverb.ReplaySample(info=info, data=data)
    iterator = tf.data.Dataset.from_tensors(sample).repeat().batch(batch_size).as_numpy_iterator()
    return distributed_utils.multi_device_put(iterator, jax.local_devices())


def priority_tables_fn(
    env_spec: worlds.EnvironmentSpec, state_spec: worlds.TreeSpec, sequence_length: int
) -> List[reverb.Table]:
    """Constructs a Reverb table containing sequences as the data primitive.

    Args:
        spec: Environment specification.
        sequence_length: Length of the sequences sampled from the dataset.

    Returns:
        List of Reverb tables containing a single sequence-based Reverb table.
    """
    signature = reverb_adders.SequenceAdder.signature(env_spec, state_spec, sequence_length=sequence_length)
    return [
        reverb.Table(
            name=_TABLE,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=100,
            rate_limiter=reverb.rate_limiters.MinSize(10),
            signature=signature,
        )
    ]


class LearnerUpdateTest(absltest.TestCase):
    """Test cases for `LearnerUpdate`."""

    def test_basic_api(self):
        game = games.RockPaperScissors(num_stages=10)
        env_spec = spec_utils.make_game_specs(game)[0]
        sequence_length = 4
        batch_size = 2
        devices = jax.local_devices()
        random_key = jax.random.PRNGKey(42)
        _, loss_graph, _, state_spec = build_computational_graphs(spec_utils.make_game_specs(game)[0])
        optimizer = optax.chain(optax.clip_by_global_norm(40.0), optax.adam(3e-4))
        data_iterator = sequence_dataset_from_spec(env_spec, state_spec, sequence_length, batch_size, devices=devices)

        updater = learner_update.LearnerUpdate(
            loss_fn=loss_graph,
            optimizer=optimizer,
            data_iterator=data_iterator,
            random_key=random_key,
            devices=devices,
        )

        state = updater.save()
        updater.restore(state)
        updater.update()
        updater.update()


if __name__ == "__main__":
    absltest.main()
