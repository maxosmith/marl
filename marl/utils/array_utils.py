"""Array operations and utility functions."""

from typing import Union

import chex
import jax
import jax.numpy as jnp
import numpy as np

from marl import types


def zeros_like(x: Union[np.ndarray, int, float, np.number]):
  """Returns a zero-filled object of the same (d)type and shape as the input.

  The difference between this and `np.zeros_like()` is that this works well
  with `np.number`, `int`, `float`, and `jax.Array` objects without
  converting them to `np.ndarray`s.

  Args:
      x: The object to replace with 0s.

  Returns:
      A zero-filed object of the same (d)type and shape as the input.
  """
  if isinstance(x, (int, float, np.number)):
    return type(x)(0)
  elif isinstance(x, jax.Array):
    return jnp.zeros_like(x)
  elif isinstance(x, np.ndarray):
    return np.zeros_like(x)
  else:
    raise ValueError(f"Input ({type(x)}) must be either a numpy array, an int, or a float.")


def one_hot(x: types.Array, num_classes: int) -> types.Array:
  """One-hot encodes an array.

  Args:
    x: Array to encode.
    num_classes: Number of classes.

  Returns:
    Array with additional final axis containing one-hot encodings.
  """
  op_lib = np if isinstance(x, np.ndarray) else jnp
  return op_lib.eye(num_classes)[x]


def broadcast_concat(x: types.Array, y: types.Array) -> types.Array:
  """Broadcast and concatenate y onto the end of x.

  Args:
    x:
    y:

  Returns:

  """
  if type(x) != type(y):
    raise ValueError("Arrays must both be numpy or jax arrays.")
  if len(y.shape) >= len(x.shape):
    raise ValueError(f"The rank of y rank ({len(y.shape)}) must be less than x ({len(x.shape)}).")
  op_lib = np if isinstance(x, np.ndarray) else jnp

  if not y.shape:
    y = y[None]

  if len(x.shape) != len(y.shape):
    y = op_lib.broadcast_to(y, x.shape[:-1] + y.shape)
  return op_lib.concatenate([x, y], axis=-1)
