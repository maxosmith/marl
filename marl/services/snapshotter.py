"""Utility classes for snapshotting models."""
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

from marl import _types
from marl.services.interfaces import variable_source_interface, worker_interface
from marl.utils import file_utils, signals

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
    params: Optional[_types.Params] = None
    variable_source_keys: Optional[Sequence[str]] = None


class Snapshotter(worker_interface.WorkerInterface):
    """Periodically fetches new version of params and saves them to disk."""

    def __init__(
        self,
        variable_source: variable_source_interface.VariableSourceInterface,
        snapshot_templates: Dict[str, Snapshot],
        directory: str,
        max_to_keep: Optional[int] = None,
        save_frequency: int = 5,
    ):
        """Initializes an instance of `Snapshotter`.

        Args:
            variable_source: Source of the snapshot parameters.
            snapshot_templates: Snapshots templates that are missing `params`.
                These are used to inform the snapshotter of the meta-data used
                to build `Snapshot`s (the non-`param` fields).
            directory: Directory that snapshots are stored in.
            max_to_keep: Maximum number of each snapshot to keep on disk.
            snapshot_frequency: Frequency, in minutes, to save a snapshot.
        """
        self._variable_source = variable_source
        self._snapshot_templates = snapshot_templates
        self._path = directory
        self._max_to_keep = max_to_keep
        self._snapshot_paths: Optional[Dict[str, str]] = None
        self._save_frequency = save_frequency
        self._stop = False

    def _signal_handler(self):
        """Handle preemption signal. Note that this must happen in the main thread."""
        logging.info("Caught SIGTERM: forcing models save.")
        # TODO(maxsmith): Cannot wait on saving if resources are dead.
        # self._save()
        self._stop = True

    def _save(self):
        if not self._snapshot_paths:
            # Lazy discovery of already existing snapshots.
            self._snapshot_paths = os.listdir(self._path)
            self._snapshot_paths.sort(reverse=True)

        snapshot_location = os.path.join(self._path, time.strftime("%Y%m%d-%H%M%S"))
        if self._snapshot_paths and self._snapshot_paths[0] == snapshot_location:
            logging.info("Snapshot for the current time already exists.")
            return
        self._snapshot_paths.insert(0, snapshot_location)
        logging.info("Saving snapshot: %s", snapshot_location)

        # Gather all snapshots to ensure they are close as possible in version.
        snapshots = self._snapshot_templates
        for name, snapshot in snapshots.items():
            snapshots[name].params = self._variable_source.get_variables(snapshot.variable_source_keys)

        # Save snapshots to disk.
        for name, snapshot in snapshots.items():
            save_to_path(os.path.join(snapshot_location, name), snapshot)

        # Delete any excess snapshots.
        while self._max_to_keep and len(self._snapshot_paths) > self._max_to_keep:
            file_utils.rm_dir(os.path.join(self._path, self._snapshot_paths.pop()))

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


class PrioritySnapshotter:
    """Saves snapshots to disk with priorities.

    Maintains the top `max_to_keep` snapshots on disk, according to their priority.
    Notably, unlike Snapshotter, this service is not a worker and expects explicit
    requests to save.
    """

    def __init__(
        self,
        snapshot_template: Snapshot,
        directory: str,
        max_to_keep: Optional[int] = None,
    ):
        """Initializes an instance of `PrioritySnapshotter`.

        Args:
            variable_source: Source of the snapshot parameters.
            snapshot_templates: Snapshots templates that are missing `params`.
                These are used to inform the snapshotter of the meta-data used
                to build `Snapshot`s (the non-`param` fields).
            directory: Directory that snapshots are stored in.
            max_to_keep: Maximum number of each snapshot to keep on disk.
            snapshot_frequency: Frequency, in minutes, to save a snapshot.
        """
        self._snapshot_template = snapshot_template
        self._path = directory
        self._max_to_keep = max_to_keep
        self._snapshot_paths: Optional[Dict[str, str]] = None
        self._snapshots: Sequence[Tuple[float, str]] = []  # Heap.

    def save(self, priority: float, params):
        if not self._snapshot_paths:
            # Lazy discovery of already existing snapshots.
            self._snapshot_paths = os.listdir(self._path)
            self._snapshot_paths.sort(reverse=True)

        snapshot_location = os.path.join(self._path, f'{time.strftime("%Y%m%d-%H%M%S")}_{priority}')
        if self._snapshot_paths and self._snapshot_paths[0] == snapshot_location:
            logging.info("Snapshot for the current time already exists.")
            return

        # Save if aren't at capacity yet.
        if len(self._snapshots) < self._max_to_keep:
            heapq.heappush(self._snapshots, (priority, snapshot_location))
            self._save(snapshot_location=snapshot_location, params=params)
            return

        # Otherwise, we need candidate to be better than the worst saved.
        baseline = self._snapshots[0][0]
        if priority > baseline:
            to_evict = heapq.heappushpop(self._snapshots, (priority, snapshot_location))
            self._save(snapshot_location=snapshot_location, params=params)
            self._delete(to_evict[1])
            return

    def _save(self, snapshot_location, params):
        """Saves a new snapshot."""
        self._snapshot_paths.insert(0, snapshot_location)
        logging.info("Saving snapshot: %s", snapshot_location)
        snapshot = self._snapshot_template
        snapshot.params = params
        save_to_path(snapshot_location, snapshot)

    def _delete(self, to_evict: str):
        """Deletes an evicted snapshot."""
        logging.info("Deleteing sanpshot: %s", to_evict)
        file_utils.rm_dir(to_evict)


def restore_from_path(ckpt_dir: str) -> Snapshot:
    """Restore the state stored in ckpt_dir."""
    # Determine if the snapshot was saved as a bundle or unbundled.
    if os.path.exists(os.path.join(ckpt_dir, _ARRAY_NAME)):
        return restore_tree(ckpt_dir, _ARRAY_NAME, _EXEMPLAR_NAME)
    else:
        return Snapshot(
            ctor=restore_tree(ckpt_dir, _CTOR_NAME, _CTOR_TREE_DEF_NAME),
            ctor_kwargs=restore_tree(ckpt_dir, _CTOR_KWARGS_NAME, _CTOR_KWARGS_TREE_DEF_NAME),
            trace_kwargs=restore_tree(ckpt_dir, _TRACE_KWARGS_NAME, _TRACE_KWARGS_TREE_DEF_NAME),
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
    save_tree(ckpt_dir, snapshot.ctor_kwargs, _CTOR_KWARGS_NAME, _CTOR_KWARGS_TREE_DEF_NAME)
    save_tree(ckpt_dir, snapshot.trace_kwargs, _TRACE_KWARGS_NAME, _TRACE_KWARGS_TREE_DEF_NAME)
    save_tree(ckpt_dir, snapshot.params, _PARAMS_NAME, _PARAMS_TREE_DEF_NAME)


def save_tree(ckpt_dir: str, data: Any, array_name: str, tree_name: str):
    """Save the state in ckpt_dir."""

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

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
