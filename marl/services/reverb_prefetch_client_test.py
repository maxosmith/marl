"""Test cases for `reverb_prefetch_client`. """
import numpy as np
import reverb
from absl.testing import absltest, parameterized

from marl.services import reverb_prefetch_client


def mock_replay_iter(batch_size: int, x: int) -> reverb.ReplaySample:
    """Mock replay buffer data iterator.

    Args:
        batch_size: Batch size.
        x: Constant to fill mock data with.

    Yields:
        Replay sample full of `x`. Each field has shape [Device, Batch].
    """
    device_dim_size = 1
    shape = (device_dim_size, batch_size)
    dummy_data = np.full(shape, x)
    while True:
        yield reverb.ReplaySample(
            info=reverb.SampleInfo(
                key=dummy_data,
                probability=dummy_data,
                table_size=dummy_data,
                priority=dummy_data,
                times_sampled=dummy_data,
            ),
            data=dummy_data,
        )


class ReverbPrefetchMixtureClientTest(parameterized.TestCase):
    """Test cases for the `ReverbPrefetchMixtureClient` class."""

    def test_cache(self):
        """Tests cache maintenance of partial used batches."""
        train_iter = mock_replay_iter(128, 1)
        plan_iter = mock_replay_iter(128, 2)
        batch_dim = 1

        client = reverb_prefetch_client.ReverbPrefetchMixtureClient(
            iterators=[train_iter, plan_iter],
            weights=[0.25, 0.75],
            batch_size=128,
        )

        sample: reverb.ReplaySample = next(client)
        self.assertEqual(128, sample.info.probability.shape[batch_dim])
        self.assertEqual(96, client._cache[0].info.probability.shape[batch_dim])
        self.assertEqual(32, client._cache[1].info.probability.shape[batch_dim])

        sample: reverb.ReplaySample = next(client)
        self.assertEqual(128, sample.info.probability.shape[batch_dim])
        self.assertEqual(64, client._cache[0].info.probability.shape[batch_dim])
        self.assertEqual(64, client._cache[1].info.probability.shape[batch_dim])

        sample: reverb.ReplaySample = next(client)
        self.assertEqual(128, sample.info.probability.shape[batch_dim])
        self.assertEqual(32, client._cache[0].info.probability.shape[batch_dim])
        self.assertEqual(96, client._cache[1].info.probability.shape[batch_dim])


if __name__ == "__main__":
    absltest.main()
