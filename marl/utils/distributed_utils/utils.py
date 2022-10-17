from typing import Callable, Optional, Sequence, TypeVar

import jax
import jax.numpy as jnp

from marl import _types

F = TypeVar("F", bound=Callable)
T = TypeVar("T", bound=_types.Tree)


def get_from_first_device(tree: T, as_numpy: bool = True) -> T:
    """Gets the first array of a tree of `jax.pxla.ShardedDeviceArray`s.

    Args:
        tree: A tree of `jax.pxla.ShardedDeviceArray`s.
        as_numpy: If `True` then each `DeviceArray` that is retrieved is transformed
        (and copied if not on the host machine) into a `np.ndarray`.

    Returns:
        The first array of a tree of `jax.pxla.ShardedDeviceArray`s. Note that if
        `as_numpy=False` then the array will be a `DeviceArray` (which will live on
        the same device as the sharded device array). If `as_numpy=True` then the
        array will be copied to the host machine and converted into a `np.ndarray`.
    """
    zeroth_tree = jax.tree_map(lambda x: x[0], tree)
    return jax.device_get(zeroth_tree) if as_numpy else zeroth_tree


def replicate_on_all_devices(tree: T, devices: Optional[Sequence[jax.xla.Device]] = None) -> T:
    """Replicate tree on all available devices."""
    devices = devices or jax.local_devices()
    return jax.device_put_sharded([tree] * len(devices), devices)
