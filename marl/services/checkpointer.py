"""Utility classes for Checkpointting models."""
import dataclasses
import heapq
import os
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import chex
import cloudpickle
import numpy as np
import tree
from absl import logging
from ml_collections import config_dict

from marl import types
from marl.services.interfaces import variable_source_interface, worker_interface
from marl.utils import file_utils, signals

# Bundled Checkpoint file names.
_ARRAY_NAME = "array_nest"
_EXEMPLAR_NAME = "tree_exemplar"
# Unbundled Checkpoint file names.
_CTOR_NAME = "ctor"
_CTOR_TREE_DEF_NAME = "ctor_tree_def"
_CTOR_KWARGS_NAME = "ctor_kwargs"
_CTOR_KWARGS_TREE_DEF_NAME = "ctor_kwargs_tree_def"
_TRACE_KWARGS_NAME = "trace_kwargs"
_TRACE_KWARGS_TREE_DEF_NAME = "trace_kwargs_tree_def"
_PARAMS_NAME = "params"
_PARAMS_TREE_DEF_NAME = "params_tree_def"


@dataclasses.dataclass
class Checkpoint:
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
          the variables hosted by the `Checkpointer`'s `variable_source`.
  """

  ctor: Callable[..., Any]
  ctor_kwargs: Mapping[str, Any]
  trace_kwargs: Optional[Mapping[str, Any]] = None
  params: Optional[types.Params] = None
  variable_source_keys: Optional[Sequence[str]] = None


class Checkpointer(worker_interface.WorkerInterface):
  """Periodically fetches new version of params and saves them to disk."""

  def __init__(
      self,
      variable_source: variable_source_interface.VariableSourceInterface,
      Checkpoint_templates: Dict[str, Checkpoint],
      directory: str,
      max_to_keep: Optional[int] = None,
      save_frequency: int = 5,
  ):
    """Initializes an instance of `Checkpointer`.

    Args:
        variable_source: Source of the Checkpoint parameters.
        Checkpoint_templates: Checkpoints templates that are missing `params`.
            These are used to inform the Checkpointer of the meta-data used
            to build `Checkpoint`s (the non-`param` fields).
        directory: Directory that Checkpoints are stored in.
        max_to_keep: Maximum number of each Checkpoint to keep on disk.
        Checkpoint_frequency: Frequency, in minutes, to save a Checkpoint.
    """
    self._variable_source = variable_source
    self._Checkpoint_templates = Checkpoint_templates
    self._path = directory
    self._max_to_keep = max_to_keep
    self._Checkpoint_paths: Optional[Dict[str, str]] = None
    self._save_frequency = save_frequency
    self._stop = False

  def _signal_handler(self):
    """Handle preemption signal. Note that this must happen in the main thread."""
    logging.info("Caught SIGTERM: forcing models save.")
    # TODO(maxsmith): Cannot wait on saving if resources are dead.
    # self._save()
    self._stop = True

  def _save(self):
    if not self._Checkpoint_paths:
      # Lazy discovery of already existing Checkpoints.
      self._Checkpoint_paths = file_utils.get_subdirs(self._path)
      self._Checkpoint_paths.sort(reverse=True)
      logging.info(self._Checkpoint_paths)

    Checkpoint_location = os.path.join(self._path, time.strftime("%Y%m%d-%H%M%S"))
    if self._Checkpoint_paths and self._Checkpoint_paths[0] == Checkpoint_location:
      logging.info("Checkpoint for the current time already exists.")
      return
    self._Checkpoint_paths.insert(0, Checkpoint_location)
    logging.info("Saving Checkpoint: %s", Checkpoint_location)

    # Gather all Checkpoints to ensure they are close as possible in version.
    Checkpoints = self._Checkpoint_templates
    for name, Checkpoint in Checkpoints.items():
      Checkpoints[name].params = self._variable_source.get_variables(
          Checkpoint.variable_source_keys
      )

    # Save Checkpoints to disk.
    for name, Checkpoint in Checkpoints.items():
      save_to_path(os.path.join(Checkpoint_location, name), Checkpoint)

    # Delete any excess Checkpoints.
    while self._max_to_keep and len(self._Checkpoint_paths) > self._max_to_keep:
      file_utils.rm_dir(os.path.join(self._path, self._Checkpoint_paths.pop()))

  def run(self):
    """Runs the saver."""
    with signals.runtime_terminator(self._signal_handler):
      logging.info("Context")

      while True:
        if self._stop:
          break

        self._save()
        logging.info("Wait")
        time.sleep(self._save_frequency * 60)
    logging.info("Out of Context")

  def stop(self):
    """Manually stop the worker."""
    self._stop = True


class PriorityCheckpointer:
  """Saves Checkpoints to disk with priorities.

  Maintains the top `max_to_keep` Checkpoints on disk, according to their priority.
  Notably, unlike Checkpointer, this service is not a worker and expects explicit
  requests to save.
  """

  def __init__(
      self,
      Checkpoint_template: Checkpoint,
      directory: str,
      max_to_keep: Optional[int] = None,
  ):
    """Initializes an instance of `PriorityCheckpointer`.

    Args:
        variable_source: Source of the Checkpoint parameters.
        Checkpoint_templates: Checkpoints templates that are missing `params`.
            These are used to inform the Checkpointer of the meta-data used
            to build `Checkpoint`s (the non-`param` fields).
        directory: Directory that Checkpoints are stored in.
        max_to_keep: Maximum number of each Checkpoint to keep on disk.
        Checkpoint_frequency: Frequency, in minutes, to save a Checkpoint.
    """
    self._Checkpoint_template = Checkpoint_template
    self._path = directory
    self._max_to_keep = max_to_keep
    self._Checkpoint_paths: Optional[Dict[str, str]] = None
    self._Checkpoints: Sequence[Tuple[float, str]] = []  # Heap.

  def save(self, priority: float, params):
    if not self._Checkpoint_paths:
      # Lazy discovery of already existing Checkpoints.
      self._Checkpoint_paths = file_utils.get_subdirs(self._path)
      self._Checkpoint_paths.sort(reverse=True)

    Checkpoint_location = os.path.join(
        self._path, f'{time.strftime("%Y%m%d-%H%M%S")}_{priority}'
    )
    if self._Checkpoint_paths and self._Checkpoint_paths[0] == Checkpoint_location:
      logging.info("Checkpoint for the current time already exists.")
      return

    # Save if aren't at capacity yet.
    if len(self._Checkpoints) < self._max_to_keep:
      heapq.heappush(self._Checkpoints, (priority, Checkpoint_location))
      self._save(Checkpoint_location=Checkpoint_location, params=params)
      return

    # Otherwise, we need candidate to be better than the worst saved.
    baseline = self._Checkpoints[0][0]
    if priority > baseline:
      to_evict = heapq.heappushpop(self._Checkpoints, (priority, Checkpoint_location))
      self._save(Checkpoint_location=Checkpoint_location, params=params)
      self._delete(to_evict[1])
      return

  def _save(self, Checkpoint_location, params):
    """Saves a new Checkpoint."""
    self._Checkpoint_paths.insert(0, Checkpoint_location)
    logging.info("Saving Checkpoint: %s", Checkpoint_location)
    Checkpoint = self._Checkpoint_template
    Checkpoint.params = params
    save_to_path(Checkpoint_location, Checkpoint)

  def _delete(self, to_evict: str):
    """Deletes an evicted Checkpoint."""
    logging.info("Deleteing sanpshot: %s", to_evict)
    file_utils.rm_dir(to_evict)


def restore_from_path(ckpt_dir: str) -> Checkpoint:
  """Restore the state stored in ckpt_dir."""
  # Determine if the Checkpoint was saved as a bundle or unbundled.
  if os.path.exists(os.path.join(ckpt_dir, _ARRAY_NAME)):
    return restore_tree(ckpt_dir, _ARRAY_NAME, _EXEMPLAR_NAME)
  else:
    return Checkpoint(
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


def save_to_path_bundled(ckpt_dir: str, Checkpoint: Checkpoint):
  """Save the state in ckpt_dir as a single object."""
  save_tree(ckpt_dir, Checkpoint, _ARRAY_NAME, _EXEMPLAR_NAME)


def save_to_path(ckpt_dir: str, Checkpoint: Checkpoint):
  """Save Checkpoint components individually.

  This save creates more files to prevent corrupting the entire Checkpoint if
  a single field of the Checkpoint is deprecated. For example, if path name for
  the ctor is modified, we would be unable to salvage the params from the
  Checkpoint when they're bundled.
  """
  save_tree(ckpt_dir, Checkpoint.ctor, _CTOR_NAME, _CTOR_TREE_DEF_NAME)
  save_tree(
      ckpt_dir, Checkpoint.ctor_kwargs, _CTOR_KWARGS_NAME, _CTOR_KWARGS_TREE_DEF_NAME
  )
  save_tree(
      ckpt_dir, Checkpoint.trace_kwargs, _TRACE_KWARGS_NAME, _TRACE_KWARGS_TREE_DEF_NAME
  )
  save_tree(ckpt_dir, Checkpoint.params, _PARAMS_NAME, _PARAMS_TREE_DEF_NAME)


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
