"""Array operations and utility functions."""

from typing import Union

import jax.numpy as jnp
import numpy as np


def zeros_like(x: Union[np.ndarray, int, float, np.number]):
    """Returns a zero-filled object of the same (d)type and shape as the input.

    The difference between this and `np.zeros_like()` is that this works well
    with `np.number`, `int`, `float`, and `jax.numpy.DeviceArray` objects without
    converting them to `np.ndarray`s.

    Args:
        x: The object to replace with 0s.

    Returns:
        A zero-filed object of the same (d)type and shape as the input.
    """
    if isinstance(x, (int, float, np.number)):
        return type(x)(0)
    elif isinstance(x, jnp.DeviceArray):
        return jnp.zeros_like(x)
    elif isinstance(x, np.ndarray):
        return np.zeros_like(x)
    else:
        raise ValueError(f"Input ({type(x)}) must be either a numpy array, an int, or a float.")


def one_hot(x: np.ndarray, num_classes: int) -> np.ndarray:
    """One-hot encodes an array."""
    return np.eye(num_classes)[x]


def broadcast_concat(x: jnp.DeviceArray, y: jnp.DeviceArray) -> jnp.DeviceArray:
    """Broadcast and concatenate y onto the end of x."""
    if len(x.shape) != len(y.shape):
        y = jnp.broadcast_to(y, x.shape[:-1] + y.shape)
    return jnp.concatenate([x, y], axis=-1)
