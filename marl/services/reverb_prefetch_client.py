import logging
import queue
import threading

import jax
import reverb

from marl.rl.replay.reverb import dataset
from marl.utils import distributed_utils


class ReverbPrefetchClient:
    """Client that prefetches reverb data onto appropriate training devices."""

    def __init__(
        self,
        reverb_client: reverb.Client,
        table_name: str,
        *,
        batch_size: int,
        buffer_size: int = 5,
        num_threads: int = 1
    ):
        """Initializes a reverb prefetch client.

        Args:
            reverb_client:
            buffer_size: Number of elements to keep in the prefetch buffer.
            num_threads: Number of threads.
        """
        if buffer_size < 1:
            raise ValueError("`buffer_size` should be >= 1.")
        if num_threads < 1:
            raise ValueError("`num_prefetch_threads` should be >= 1.")

        batch_size_per_learner = batch_size // jax.process_count()
        batch_size_per_device, ragged = divmod(batch_size, jax.device_count())
        if ragged:
            raise ValueError("Learner batch size must be divisible by total number of devices!")

        self.count = 0

        # Make reverb dataset iterator.
        self._buffer = queue.Queue(maxsize=buffer_size)
        self._producer_error = []
        self._end = object()
        ds = dataset.make_reverb_dataset(
            table=table_name,
            server_address=reverb_client.server_address,
            batch_size=batch_size_per_device,
            num_parallel_calls=None,
            max_in_flight_samples_per_worker=2 * batch_size_per_learner,
        )
        self._reverb_iterator = distributed_utils.multi_device_put(ds.as_numpy_iterator(), jax.local_devices())

        for _ in range(num_threads):
            threading.Thread(target=self.prefetch, daemon=True).start()

    def prefetch(self):
        """Enqueues items from the reverb."""
        try:
            # Build a new iterable for each thread. This is crucial if working with
            # tensorflow datasets because tf.Graph objects are thread local.
            for item in self._reverb_iterator:
                self._buffer.put(item)
        except Exception as e:  # pylint: disable=broad-except
            logging.exception("Error in producer thread for %s", self._reverb_iterator)
            self.producer_error.append(e)
        finally:
            self._buffer.put(self._end)

    def ready(self):
        return not self._buffer.empty()

    def retrieved_elements(self):
        """Number of retrievals from the iterator."""
        return self.count

    def __iter__(self):
        return self

    def __next__(self):
        value = self._buffer.get()
        if value is self._end:
            if self._producer_error:
                raise self._producer_error[0] from self._producer_error[0]
            else:
                raise StopIteration
        self.count += 1
        return value
