import heapq
import os
import time
from typing import Dict, Optional, Sequence, Tuple

from absl import logging

from marl.services.snapshotter import utils
from marl.utils import file_utils


class PrioritySnapshotter:
  """Saves snapshots to disk with priorities.

  Maintains the top `max_to_keep` snapshots on disk, according to their priority.
  Notably, unlike Snapshotter, this service is not a worker and expects explicit
  requests to save.
  """

  def __init__(
      self,
      snapshot_template: utils.Snapshot,
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
      self._snapshot_paths = file_utils.get_subdirs(self._path)
      self._snapshot_paths.sort(reverse=True)

    snapshot_location = os.path.join(
        self._path, f'{time.strftime("%Y%m%d-%H%M%S")}_{priority}'
    )
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
    utils.save_to_path(snapshot_location, snapshot)

  def _delete(self, to_evict: str):
    """Deletes an evicted snapshot."""
    logging.info("Deleteing sanpshot: %s", to_evict)
    file_utils.rm_dir(to_evict)
