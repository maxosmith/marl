import itertools
import logging
from typing import Callable, Iterable, Iterator, NamedTuple, Optional, Sequence

import jax
import numpy as np
import tree

from marl import _types


class PrefetchingSplit(NamedTuple):
    host: _types.Tree
    device: _types.Tree


_SplitFunction = Callable[[_types.Tree], PrefetchingSplit]


def device_put(
    iterable: Iterable[_types.Tree],
    device: jax.xla.Device,
    split_fn: Optional[_SplitFunction] = None,
):
    """Returns iterator that samples an item and places it on the device."""

    return PutToDevicesIterable(iterable=iterable, pmapped_user=False, devices=[device], split_fn=split_fn)


def multi_device_put(
    iterable: Iterable[_types.Tree],
    devices: Sequence[jax.xla.Device],
    split_fn: Optional[_SplitFunction] = None,
):
    """Returns iterator that, per device, samples an item and places on device."""

    return PutToDevicesIterable(iterable=iterable, pmapped_user=True, devices=devices, split_fn=split_fn)


class PutToDevicesIterable(Iterable[_types.Tree]):
    """Per device, samples an item from iterator and places on device.
    if pmapped_user:
      Items from the resulting generator are intended to be used in a pmapped
      function. Every element is a ShardedDeviceArray or (nested) Python container
      thereof. A single next() call to this iterator results in len(devices)
      calls to the underlying iterator. The returned items are put one on each
      device.
    if not pmapped_user:
      Places a sample from the iterator on the given device.
    Yields:
      If no split_fn is specified:
        DeviceArray/ShardedDeviceArray or (nested) Python container thereof
        representing the elements of shards stacked together, with each shard
        backed by physical device memory specified by the corresponding entry in
        devices.
      If split_fn is specified:
        PrefetchingSplit where the .host element is a stacked numpy array or
        (nested) Python contained thereof. The .device element is a
        DeviceArray/ShardedDeviceArray or (nested) Python container thereof.
    Raises:
      StopIteration: if there are not enough items left in the iterator to place
        one sample on each device.
      Any error thrown by the iterable_function. Note this is not raised inside
        the producer, but after it finishes executing.
    """

    def __init__(
        self,
        iterable: Iterable[_types.Tree],
        pmapped_user: bool,
        devices: Sequence[jax.xla.Device],
        split_fn: Optional[_SplitFunction] = None,
    ):
        """Constructs PutToDevicesIterable.
        Args:
          iterable: A python iterable. This is used to build the python prefetcher.
            Note that each iterable should only be passed to this function once as
            iterables aren't thread safe.
          pmapped_user: whether the user of data from this iterator is implemented
            using pmapping.
          devices: Devices used for prefecthing.
          split_fn: Optional function applied to every element from the iterable to
            split the parts of it that will be kept in the host and the parts that
            will sent to the device.
        Raises:
          ValueError: If devices list is empty, or if pmapped_use=False and more
            than 1 device is provided.
        """
        self.num_devices = len(devices)
        if self.num_devices == 0:
            raise ValueError("At least one device must be specified.")
        if (not pmapped_user) and (self.num_devices != 1):
            raise ValueError(
                "User is not implemented with pmapping but len(devices) "
                f"= {len(devices)} is not equal to 1! Devices given are:"
                f"\n{devices}"
            )

        self.iterable = iterable
        self.pmapped_user = pmapped_user
        self.split_fn = split_fn
        self.devices = devices
        self.iterator = iter(self.iterable)

    def __iter__(self) -> Iterator[_types.Tree]:
        # It is important to structure the Iterable like this, because in
        # JustPrefetchIterator we must build a new iterable for each thread.
        # This is crucial if working with tensorflow datasets because tf.Graph
        # objects are thread local.
        self.iterator = iter(self.iterable)
        return self

    def __next__(self) -> _types.Tree:
        try:
            if not self.pmapped_user:
                item = next(self.iterator)
                if self.split_fn is None:
                    return jax.device_put(item, self.devices[0])
                item_split = self.split_fn(item)
                return PrefetchingSplit(host=item_split.host, device=jax.device_put(item_split.device, self.devices[0]))

            items = itertools.islice(self.iterator, self.num_devices)
            items = tuple(items)
            if len(items) < self.num_devices:
                raise StopIteration
            if self.split_fn is None:
                return jax.device_put_sharded(tuple(items), self.devices)
            else:
                # ((host: x1, device: y1), ..., (host: xN, device: yN)).
                items_split = (self.split_fn(item) for item in items)
                # (host: (x1, ..., xN), device: (y1, ..., yN)).
                split = tree.map_structure_up_to(PrefetchingSplit(None, None), lambda *x: x, *items_split)

                return PrefetchingSplit(
                    host=np.stack(split.host), device=jax.device_put_sharded(split.device, self.devices)
                )

        except StopIteration:
            raise

        except Exception:  # pylint: disable=broad-except
            logging.exception("Error for %s", self.iterable)
            raise
