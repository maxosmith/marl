"""Utility classes for snapshotting models."""
import dataclasses
import os
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import chex
import cloudpickle
import numpy as np
import tree
from absl import logging

from marl import _types
from marl.services.interfaces import variable_source_interface, worker_interface
from marl.utils import file_utils, signals

_ARRAY_NAME = "array_nest"
_EXEMPLAR_NAME = "tree_exemplar"


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
        """Initializes an instance3 of `Snapshotter`.

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
        self._snapshot_paths: Optional[List[str]] = None
        self._save_frequency = save_frequency

    def _signal_handler(self):
        """Handle preemption signal. Note that this must happen in the main thread."""
        logging.info("Caught SIGTERM: forcing models save.")
        self._save()

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
        snapshot_paths = []
        for name, snapshot in snapshots.items():
            snapshots[name].params = self._variable_source.get_variables(snapshot.variable_source_keys)
            snapshot_paths.append(os.path.join(snapshot_location, name))

        # Save snapshots to disk.
        for snapshot, path in zip(snapshots, snapshot_paths):
            save_to_path(path, snapshot)

        # Delete any excess snapshots.
        while self._max_to_keep and len(self._snapshot_paths) > self._max_to_keep:
            file_utils.rm_dir(os.path.join(self._path, self._snapshot_paths.pop()))

    def run(self):
        """Runs the saver."""
        with signals.runtime_terminator(self._signal_handler):
            while True:
                self._save()
                time.sleep(self._save_frequency * 60)


def restore_from_path(ckpt_dir: str) -> Snapshot:
    """Restore the state stored in ckpt_dir."""
    array_path = os.path.join(ckpt_dir, _ARRAY_NAME)
    exemplar_path = os.path.join(ckpt_dir, _EXEMPLAR_NAME)

    with open(exemplar_path, "rb") as f:
        exemplar = cloudpickle.load(f)

    with open(array_path, "rb") as f:
        files = np.load(f, allow_pickle=True)
        flat_state = [files[key] for key in files.files]
    unflattened_tree = tree.unflatten_as(exemplar, flat_state)

    def maybe_convert_to_python(value, numpy):
        return value if numpy else value.item()

    return tree.map_structure(maybe_convert_to_python, unflattened_tree, exemplar)


def save_to_path(ckpt_dir: str, snapshot: Snapshot):
    """Save the state in ckpt_dir."""

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    is_numpy = lambda x: isinstance(x, (np.ndarray, chex.Array))
    flat_state = tree.flatten(snapshot)
    nest_exemplar = tree.map_structure(is_numpy, snapshot)

    array_path = os.path.join(ckpt_dir, _ARRAY_NAME)
    logging.info("Saving flattened array nest to %s", array_path)

    def _disabled_seek(*_):
        raise AttributeError("seek() is disabled on this object.")

    with open(array_path, "wb") as f:
        setattr(f, "seek", _disabled_seek)
        np.savez(f, *flat_state)

    exemplar_path = os.path.join(ckpt_dir, _EXEMPLAR_NAME)
    logging.info("Saving nest exemplar to %s", exemplar_path)
    with open(exemplar_path, "wb") as f:
        cloudpickle.dump(nest_exemplar, f)
