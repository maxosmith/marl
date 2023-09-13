"""Tests for `snapshotter`."""
import dataclasses
import os.path as osp
import tempfile
import time
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from marl import types
from marl.services.snapshotter import priority_snapshotter, utils
from marl.utils import file_utils, tree_utils

# `Snapshotter`` should only modify the `params` field of a `Snapshot`.
_DUMMY_SNAPSHOT = utils.Snapshot(ctor=None, ctor_kwargs=None)


@dataclasses.dataclass
class _Linear:
  """Linear function (y=m*x+b)."""

  m: jax.Array
  b: jax.Array

  def __call__(self, x: jax.Array) -> jax.Array:
    """Forward pass."""
    return jnp.add(jnp.multiply(self.m, x), self.b)


class PrioritySnapshotterTest(parameterized.TestCase):

  def setUp(self):
    """Set-up performed for each test case."""
    self.tmp_dir = tempfile.TemporaryDirectory()

  def tearDown(self):
    """Tear-down performed after each test case."""
    self.tmp_dir.cleanup()

  @parameterized.parameters(
      dict(parameters=(0.0,)),
      dict(parameters=(0.0, 1.0)),
      dict(parameters=(dict(x=1, y=(1, 3.0)), dict(x=2, y=(4, 9.0)))),
  )
  def test_snapshot_params(self, parameters: Sequence[types.Tree]):
    """Test saving and loading `Snapshot` parameters."""
    snap = priority_snapshotter.PrioritySnapshotter(
        snapshot_template=_DUMMY_SNAPSHOT,
        directory=self.tmp_dir.name,
    )

    # Save all snapshots.
    for param_i, param in enumerate(parameters):
      snap.save(priority=param_i, params=param)

    # Verify the snapshots are all saved, and that they are ordered
    # correctly in time.
    subdirs = sorted(file_utils.get_subdirs(self.tmp_dir.name))
    for subdir, (expected_priority, expected_params) in zip(
        subdirs, enumerate(parameters)
    ):
      priority = int(subdir.split("_")[-1])
      self.assertEqual(priority, expected_priority)
      tree_utils.assert_equals(
          utils.restore_from_path(osp.join(self.tmp_dir.name, subdir)).params,
          expected_params,
      )

  @parameterized.parameters(1, 3)
  def test_max_to_keep(self, max_to_keep):
    """Test saving only `max_to_keep` recent snapshots."""
    parameters = (0.0, 1.0, 2.0, 3.0)
    priorities = (1, 4, 2, 3)
    prio_to_params = {prio: param for prio, param in zip(priorities, parameters)}

    snap = priority_snapshotter.PrioritySnapshotter(
        snapshot_template=_DUMMY_SNAPSHOT,
        directory=self.tmp_dir.name,
    )

    # Save all snapshots.
    for priority, params in zip(priorities, parameters):
      snap.save(priority=priority, params=params)

    subdirs = sorted(file_utils.get_subdirs(self.tmp_dir.name))
    prio_to_subdir = {int(x.split("_")[-1]): x for x in subdirs}

    # Check that highest priority saves are accurate.
    highest_prio = 4
    for _ in range(max_to_keep):
      self.assertIn(highest_prio, prio_to_subdir)
      save_path = osp.join(self.tmp_dir.name, prio_to_subdir[highest_prio])
      tree_utils.assert_equals(
          utils.restore_from_path(save_path).params,
          prio_to_params[highest_prio],
      )

      highest_prio -= 1

  @parameterized.parameters(
      dict(
          ctor=_Linear,
          ctor_kwargs=dict(m=2.0, b=5.0),
          trace_kwargs=dict(x=1.0),
          expected_output=jnp.asarray(7.0, dtype=jnp.float32),
      ),
      dict(
          ctor=_Linear,
          ctor_kwargs=dict(m=2.0, b=5.0),
          trace_kwargs=dict(x=jnp.asarray([1.0, 2.0], dtype=jnp.float32)),
          expected_output=jnp.asarray([7.0, 9.0], dtype=jnp.float32),
      ),
      dict(
          ctor=_Linear,
          ctor_kwargs=dict(m=2.0, b=4.0),
          trace_kwargs=dict(x=jnp.asarray([1.0, 2.0], dtype=jnp.float32)),
          expected_output=jnp.asarray([6.0, 8.0], dtype=jnp.float32),
      ),
  )
  def test_snapshot_ctor(
      self,
      ctor: Any,
      ctor_kwargs: Mapping[str, Any],
      trace_kwargs: Mapping[str, Any],
      expected_output: jax.Array,
  ):
    """Tests the snapshot ctor, ctor_kwargs, and trace_kwargs save/load."""
    snap = priority_snapshotter.PrioritySnapshotter(
        snapshot_template=utils.Snapshot(
            ctor=ctor,
            ctor_kwargs=ctor_kwargs,
            trace_kwargs=trace_kwargs,
        ),
        directory=self.tmp_dir.name,
    )
    snap.save(1, None)

    subdirs = sorted(file_utils.get_subdirs(self.tmp_dir.name))
    self.assertLen(subdirs, 1)
    snapshot = utils.restore_from_path(osp.join(self.tmp_dir.name, subdirs[0]))

    loaded_op = snapshot.ctor(**snapshot.ctor_kwargs)
    loaded_op = jax.jit(loaded_op.__call__)
    output = loaded_op(**snapshot.trace_kwargs)
    tree_utils.assert_equals(output, expected_output)


if __name__ == "__main__":
  absltest.main()
