"""Utility functions for saving snapshots."""
import dataclasses
import os
from typing import Any, Callable, Mapping, Optional, Sequence

import chex
import cloudpickle
import numpy as np
import tree
from absl import logging
from ml_collections import config_dict

from marl import types

# Bundled snapshot file names.
_ARRAY_NAME = "array_nest"
_EXEMPLAR_NAME = "tree_exemplar"
# Unbundled snapshot file names.
_CTOR_NAME = "ctor"
_CTOR_TREE_DEF_NAME = "ctor_tree_def"
_CTOR_KWARGS_NAME = "ctor_kwargs"
_CTOR_KWARGS_TREE_DEF_NAME = "ctor_kwargs_tree_def"
_TRACE_KWARGS_NAME = "trace_kwargs"
_TRACE_KWARGS_TREE_DEF_NAME = "trace_kwargs_tree_def"
_PARAMS_NAME = "params"
_PARAMS_TREE_DEF_NAME = "params_tree_def"


@dataclasses.dataclass
class Snapshot:
  """Stores all data necessary to save and reload a model to disk.

  Attributes:
      ctor: Constructor used to instantiate an instance of the model.
          This must be accessible from the global namespace, so prefer
          not using partial/lambda to close parameters.
      ctor_kwargs: Keyword arguments used when calling `ctor`.
      params: The model's parameters.
      trace_kwargs: Placeholder inputs that would be given to the model's
          call method that ensures that the inputs can be traced and that
          the shapes throughout the model can be built.
      variable_source_keys: Optional list of names identifying a subset of
          the variables hosted by the `Snapshotter`'s `variable_source`.
  """

  ctor: Callable[..., Any]
  ctor_kwargs: Mapping[str, Any]
  trace_kwargs: Optional[Mapping[str, Any]] = None
  params: Optional[types.Params] = None
  variable_source_keys: Optional[Sequence[str]] = None


def restore_from_path(ckpt_dir: str) -> Snapshot:
  """Restore the state stored in ckpt_dir."""
  # Determine if the snapshot was saved as a bundle or unbundled.
  if os.path.exists(os.path.join(ckpt_dir, _ARRAY_NAME)):
    return restore_tree(ckpt_dir, _ARRAY_NAME, _EXEMPLAR_NAME)
  else:
    return Snapshot(
        ctor=restore_tree(ckpt_dir, _CTOR_NAME, _CTOR_TREE_DEF_NAME),
        ctor_kwargs=restore_tree(
            ckpt_dir, _CTOR_KWARGS_NAME, _CTOR_KWARGS_TREE_DEF_NAME
        ),
        trace_kwargs=restore_tree(
            ckpt_dir, _TRACE_KWARGS_NAME, _TRACE_KWARGS_TREE_DEF_NAME
        ),
        params=restore_tree(ckpt_dir, _PARAMS_NAME, _PARAMS_TREE_DEF_NAME),
    )


def restore_tree(ckpt_dir: str, name: str, tree_def_name: str) -> Any:
  """Restore the state stored in ckpt_dir."""
  array_path = os.path.join(ckpt_dir, name)
  exemplar_path = os.path.join(ckpt_dir, tree_def_name)

  with open(exemplar_path, "rb") as f:
    exemplar = cloudpickle.load(f)

  with open(array_path, "rb") as f:
    files = np.load(f, allow_pickle=True)
    flat_state = [files[key] for key in files.files]
  unflattened_tree = tree.unflatten_as(exemplar, flat_state)

  def maybe_convert_to_python(value, numpy):
    return value if numpy else value.item()

  return tree.map_structure(maybe_convert_to_python, unflattened_tree, exemplar)


def save_to_path_bundled(ckpt_dir: str, snapshot: Snapshot):
  """Save the state in ckpt_dir as a single object."""
  save_tree(ckpt_dir, snapshot, _ARRAY_NAME, _EXEMPLAR_NAME)


def save_to_path(ckpt_dir: str, snapshot: Snapshot):
  """Save snapshot components individually.

  This save creates more files to prevent corrupting the entire snapshot if
  a single field of the snapshot is deprecated. For example, if path name for
  the ctor is modified, we would be unable to salvage the params from the
  snapshot when they're bundled.
  """
  save_tree(ckpt_dir, snapshot.ctor, _CTOR_NAME, _CTOR_TREE_DEF_NAME)
  save_tree(
      ckpt_dir, snapshot.ctor_kwargs, _CTOR_KWARGS_NAME, _CTOR_KWARGS_TREE_DEF_NAME
  )
  save_tree(
      ckpt_dir, snapshot.trace_kwargs, _TRACE_KWARGS_NAME, _TRACE_KWARGS_TREE_DEF_NAME
  )
  save_tree(ckpt_dir, snapshot.params, _PARAMS_NAME, _PARAMS_TREE_DEF_NAME)


def save_tree(ckpt_dir: str, data: Any, array_name: str, tree_name: str):
  """Save the state in ckpt_dir."""

  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

  def _config_to_dict(x):
    """Convert all config dicts to nested dicts so they're tree-like."""
    return x.to_dict() if isinstance(x, config_dict.ConfigDict) else x

  data = tree.map_structure(_config_to_dict, data)

  def _is_numpy(x):
    """Check if item is a numpy-like array."""
    return isinstance(x, (np.ndarray, chex.Array))

  flat_state = tree.flatten(data)
  nest_exemplar = tree.map_structure(_is_numpy, data)

  array_path = os.path.join(ckpt_dir, array_name)
  logging.info("Saving flattened array nest to %s", array_path)

  def _disabled_seek(*_):
    """Disable the seek operation."""
    raise AttributeError("seek() is disabled on this object.")

  with open(array_path, "wb") as f:
    setattr(f, "seek", _disabled_seek)
    np.savez(f, *flat_state)

  exemplar_path = os.path.join(ckpt_dir, tree_name)
  logging.info("Saving nest exemplar to %s", exemplar_path)

  with open(exemplar_path, "wb") as f:
    cloudpickle.dump(nest_exemplar, f)
