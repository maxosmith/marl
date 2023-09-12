"""Utility classes for snapshotting models."""
import os
import time
from typing import Dict, Optional

from absl import logging

from marl.services import interfaces, variable_client
from marl.services.snapshotter import utils
from marl.utils import file_utils, signals


class Snapshotter(interfaces.WorkerInterface):
  """Periodically fetches new version of params and saves them to disk."""

  def __init__(
      self,
      variable_source: variable_client.VariableClient,
      snapshot_templates: Dict[str, utils.Snapshot],
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
      self._snapshot_paths = file_utils.get_subdirs(self._path)
      self._snapshot_paths.sort(reverse=True)
      logging.info(self._snapshot_paths)

    snapshot_location = os.path.join(self._path, time.strftime("%Y%m%d-%H%M%S"))
    if self._snapshot_paths and self._snapshot_paths[0] == snapshot_location:
      logging.info("Snapshot for the current time already exists.")
      return
    self._snapshot_paths.insert(0, snapshot_location)
    logging.info("Saving snapshot: %s", snapshot_location)

    # Gather all snapshots to ensure they are close as possible in version.
    snapshots = self._snapshot_templates
    for name, snapshot in snapshots.items():
      snapshots[name].params = self._variable_source.get_variables(
          snapshot.variable_source_keys
      )

    # Save snapshots to disk.
    for name, snapshot in snapshots.items():
      utils.save_to_path(os.path.join(snapshot_location, name), snapshot)

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
