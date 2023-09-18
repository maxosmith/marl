"""Tree, arbitrarily nested structures, utility functions."""
import operator
from typing import Any, List, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import tree

from marl import types


def to_numpy(values: types.Tree) -> types.Tree:
  def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "numpy"):  # tf.Tensor (TF2).
      return x.numpy()
    else:
      return np.asarray(x)

  return jax.tree_map(_to_numpy, values)


def _fast_map_structure(func, *structure):
  """Faster map_structure implementation which skips some error checking."""
  flat_structure = (tree.flatten(s) for s in structure)
  entries = zip(*flat_structure)
  # Arbitrarily choose one of the structures of the original sequence (the last)
  # to match the structure for the flattened sequence.
  return tree.unflatten_as(structure[-1], [func(*x) for x in entries])


def stack(sequence: Sequence[types.Tree], safe: bool = False) -> types.Tree:
  """Stacks a list of identically nested objects.

  This takes a sequence of identically nested objects and returns a single
  nested object whose ith leaf is a stacked numpy array of the corresponding
  ith leaf from each element of the sequence.

  For example, if `sequence` is:
  ```python
  [{
          'action': np.array([1.0]),
          'observation': (np.array([0.0, 1.0, 2.0]),),
          'reward': 1.0
  }, {
          'action': np.array([0.5]),
          'observation': (np.array([1.0, 2.0, 3.0]),),
          'reward': 0.0
  }, {
          'action': np.array([0.3]),1
          'observation': (np.array([2.0, 3.0, 4.0]),),
          'reward': 0.5
  }]
  ```

  Then this function will return:
  ```python
  {
      'action': np.array([....])         # array shape = [3 x 1]
      'observation': (np.array([...]),)  # array shape = [3 x 3]
      'reward': np.array([...])          # array shape = [3]
  }
  ```

  Note that the 'observation' entry in the above example has two levels of
  nesting, i.e it is a tuple of arrays.

  Args:
      sequence: a list of identically nested objects.

  Returns:
      A nested object with numpy.

  Raises:
      ValueError: If `sequence` is an empty sequence.
  """
  # Handle empty input sequences.
  if not sequence:
    raise ValueError("Input sequence must not be empty")
  map_structure = tree.map_structure if safe else _fast_map_structure

  # Default to asarray when arrays don't have the same shape to be compatible
  # with old behaviour.
  # TODO(b/169306678) make this more elegant.
  try:
    return map_structure(lambda *values: np.stack(values), *sequence)
  except ValueError:
    return map_structure(lambda *values: np.asarray(values), *sequence)


def unstack(struct: types.Tree, batch_size: int) -> List[types.Tree]:
  """Converts a struct of batched arrays to a list of structs.

  This is effectively the inverse of `stack_sequence_fields`.

  Args:
      struct: An (arbitrarily nested) structure of arrays.
      batch_size: The length of the leading dimension of each array in the struct.
      This is assumed to be static and known.

  Returns:
      A list of structs with the same structure as `struct`, where each leaf node
      is an unbatched element of the original leaf node.
  """
  return [tree.map_structure(lambda s, i=i: s[i], struct) for i in range(batch_size)]


def broadcast_structures(*args: Any) -> Any:
  """Returns versions of the arguments that give them the same nested structure.

  Any nested items in *args must have the same structure.

  Any non-nested item will be replaced with a nested version that shares that
  structure. The leaves will all be references to the same original non-nested
  item.

  If all *args are nested, or all *args are non-nested, this function will
  return *args unchanged.

  Example:
  ```
  a = ('a', 'b')
  b = 'c'
  tree_a, tree_b = broadcast_structure(a, b)
  tree_a
  > ('a', 'b')
  tree_b
  > ('c', 'c')
  ```

  Args:
      *args: A Sequence of nested or non-nested items.

  Returns:
      `*args`, except with all items sharing the same nest structure.
  """
  if not args:
    return

  reference_tree = None
  for arg in args:
    if tree.is_nested(arg):
      reference_tree = arg
      break

  # If reference_tree is None then none of args are nested and we can skip over
  # the rest of this function, which would be a no-op.
  if reference_tree is None:
    return args

  def mirror_structure(value, reference_tree):
    if tree.is_nested(value):
      # Use checktypes=True so that the types of the trees we construct aren't
      # dependent on our arbitrary choice of which nested arg to use as the
      # reference_tree.
      tree.assert_same_structure(value, reference_tree, checktypes=True)
      return value
    else:
      return tree.map_structure(lambda _: value, reference_tree)

  return tuple(mirror_structure(arg, reference_tree) for arg in args)


def assert_equals(x: types.Tree, y: types.Tree) -> None:
  """Asserts that two trees contain the same specs and structure.

  Args:
      x: a tree.
      y: a tree.

  Raises:
      ValueError: if the two structures differ.
      TypeError: if the two structures differ in their type.
  """
  tree.assert_same_structure(x, y)
  tree.map_structure(np.testing.assert_array_equal, x, y)


def assert_almost_equals(x: types.Tree, y: types.Tree, **kwargs) -> None:
  """Asserts that two trees contain the same specs and structure.

  Args:
      x: a tree.
      y: a tree.

  Raises:
      ValueError: if the two structures differ.
      TypeError: if the two structures differ in their type.
  """

  def _compare(x_, y_):
    """Comparison closing kwargs."""
    np.testing.assert_array_almost_equal(x_, y_, **kwargs)

  tree.assert_same_structure(x, y)
  tree.map_structure(_compare, x, y)


def flatten_as_dict(x: types.Tree, delimiter: str = "/") -> Mapping[str, Any]:
  """Flattens a tree into a dictionary.

  Args:
      x:
      delimiter:

  Raises:
      ValueError: if `x` is a non-tree structure.
      ValueError: if an item in `x` is not a supported type.
  """
  if isinstance(x, (int, float)) or (x is np.ndarray and x.shape == ()):
    raise ValueError("Cannot flatten a scalar.")

  def _flatten(y: types.Tree, key_prefix: str) -> Mapping[str, Any]:
    """Recursively flatten the tree."""
    if isinstance(y, (int, float)) or (hasattr(y, "shape") and y.shape == ()):
      return {key_prefix: y}

    tmp = {}
    key_prefix = key_prefix if not key_prefix else f"{key_prefix}{delimiter}"
    if isinstance(y, (np.ndarray, jax.Array, list)):
      for index, value in enumerate(y):
        tmp.update(_flatten(value, f"{key_prefix}{index}"))
    elif isinstance(y, dict):
      for key, value in y.items():
        tmp.update(_flatten(value, f"{key_prefix}{key}"))
    else:
      raise ValueError(f"Unexpected type: {type(y)}.")
    return tmp

  return _flatten(x, "")


def add_batch(struct: types.Tree, batch_size: Optional[int]):
  """Adds a batch dimension at axis 0 to all leaves."""
  broadcast = lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape)
  return jax.tree_util.tree_map(broadcast, struct)
