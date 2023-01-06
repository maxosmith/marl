"""Reverb data prefetchers."""
import logging
import queue
import threading
from typing import Iterator, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import reverb

from marl.services.replay.reverb import dataset
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
        num_threads: int = 1,
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
        self.batch_size = batch_size

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


class ReverbPrefetchMixtureClient:
    """Allows for batch-wise mixing of multiple pre-fetch clients.

    Data from the iterators is assumed to have a leading Device dimension before Batch.
    """

    def __init__(self, iterators: Sequence[Iterator], weights: Sequence[float], batch_size: int):
        """Initialize a ReverbPrefetchMixtureClient.

        Args:
            iterators: List of data iterators.
            weights: Weights for each iterator's priority.
                NOTE: Currently we mix batches by factoring the total batch-size by each weight.
            batch_size: Expected batch-size.
        """
        if len(iterators) != len(weights):
            raise ValueError("Weight must be specified for each iterator.")
        if (np.sum(weights) != 1.0) or np.any(np.array(weights) < 0):
            raise ValueError("Weights should sum to 1 and be non-negative.")
        if batch_size < 0:
            raise ValueError("Batch size must be positive.")

        # Remove unweighted iterators.
        self._iters = [itr for itr, w in zip(iterators, weights) if w]
        self._weights = [w for w in weights if w]
        self._num_iters = len(self._iters)

        self._batch_size = batch_size

        # TODO(maxsmith): Right now, we assume that batch-size is divisable by weights.
        #   This _could_ be updated to allow for higher-fidelity weights, by sampling per example.
        self._samples_per_iter = [int(w * self._batch_size) for w in self._weights]

        if np.sum(self._samples_per_iter) != batch_size:
            raise ValueError(
                f"Iter weights do not reconstruct the correct batch-size\n {batch_size=}\n {self._samples_per_iter=}"
            )

        # Cache partial samples from iterators.
        self._cache = {}

    def __iter__(self):
        return self

    def __next__(self):
        # Construct a batch by getting a sample from each iterator.
        tree_def = None
        batch_components = []
        for iter_i in range(self._num_iters):
            sample = self._get_iter_sample(iter_i)
            if not iter_i:
                tree_def = jax.tree_util.tree_structure(sample)
            # NamedTuples are reconstructed with different types of the same name, so we cannot
            # traverse these structures in parallel.
            # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/nested_structure_coder.py#L216
            sample, _ = jax.tree_util.tree_flatten(sample)
            batch_components.append(sample)

        batch = jax.tree_util.tree_map(lambda *args: jnp.concatenate(args, axis=1), *batch_components)
        batch = jax.tree_util.tree_unflatten(tree_def, batch)
        return batch

    def _maybe_call_iter(self, index: int):
        """Maybe call the iterator if we don't have enough data on it in the cache."""
        num_examples_needed = self._samples_per_iter[index]

        if index not in self._cache:
            self._call_iter(index)
        else:
            num_examples_cached = jax.tree_util.tree_leaves(self._cache[index])[0].shape[0]
            if num_examples_needed > num_examples_cached:
                self._call_iter(index)

    def _call_iter(self, index: int):
        """Call the iter and store the result in the cache."""
        batch = next(self._iters[index])

        if index not in self._cache:
            self._cache[index] = batch
        else:
            self._cache[index] = jax.tree_util.tree_map(
                lambda *args: jnp.concatenate(args, axis=1), self._cache[index], batch
            )

    def _get_iter_sample(self, index: int):
        """Get a sample from an iterator."""
        self._maybe_call_iter(index)
        size = self._samples_per_iter[index]
        sample = jax.tree_util.tree_map(lambda x: x[:, :size], self._cache[index])
        self._cache[index] = jax.tree_util.tree_map(lambda x: x[:, size:], self._cache[index])
        return sample
