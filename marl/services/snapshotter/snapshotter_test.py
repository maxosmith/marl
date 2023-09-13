"""Tests for `snapshotter`."""
import dataclasses
import os.path as osp
import tempfile
import time
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import tree
from absl.testing import absltest, parameterized

from marl import types
from marl.services import test_utils
from marl.services.snapshotter import snapshotter, utils
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


class SnapshotterTest(parameterized.TestCase):

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
  def test_snapshot_params(self, parameters: types.Tree):
    """Test saving and loading `Snapshot` parameters."""
    snap = snapshotter.Snapshotter(
        variable_source=test_utils.StubVariableSource(parameters),
        snapshot_templates={"foo": _DUMMY_SNAPSHOT},
        directory=self.tmp_dir.name,
    )

    # Save all snapshots.
    for _ in range(len(parameters)):
      snap.save()
      time.sleep(2)

    # Verify the snapshots are all saved, and that they are ordered
    # correctly in time.
    subdirs = sorted(file_utils.get_subdirs(self.tmp_dir.name))
    for subdir, expected_params in zip(subdirs, parameters):
      tree_utils.assert_equals(
          utils.restore_from_path(osp.join(self.tmp_dir.name, subdir, "foo")).params,
          expected_params,
      )

  @parameterized.parameters(1, 3)
  def test_max_to_keep(self, max_to_keep):
    """Test saving only `max_to_keep` recent snapshots."""
    parameters = (0.0, 1.0, 2.0, 3.0)
    snap = snapshotter.Snapshotter(
        variable_source=test_utils.StubVariableSource(parameters),
        snapshot_templates={"foo": _DUMMY_SNAPSHOT},
        directory=self.tmp_dir.name,
        max_to_keep=max_to_keep,
    )

    # Save all snapshots.
    for _ in range(len(parameters)):
      snap.save()
      time.sleep(2)

    # Verify the snapshots are all saved, and that they are ordered
    # correctly in time.
    subdirs = sorted(file_utils.get_subdirs(self.tmp_dir.name))
    for subdir, expected_params in zip(subdirs, parameters[-max_to_keep:]):
      tree_utils.assert_equals(
          utils.restore_from_path(osp.join(self.tmp_dir.name, subdir, "foo")).params,
          expected_params,
      )

  def test_multisave(self):
    """Test saving multiple snapshots."""
    parameters = dict(a=1, b=2)
    snap = snapshotter.Snapshotter(
        variable_source=test_utils.StubVariableSource((
            # Snapshotter will query the VariableSource for each new parameter, so
            # we need to repeat the data here twice.
            parameters,
            parameters,
        )),
        snapshot_templates={
            "a": utils.Snapshot(ctor=None, ctor_kwargs=None, variable_source_keys="a"),
            "b": utils.Snapshot(ctor=None, ctor_kwargs=None, variable_source_keys="b"),
        },
        directory=self.tmp_dir.name,
    )
    snap.save()
    subdirs = sorted(file_utils.get_subdirs(self.tmp_dir.name))
    self.assertLen(subdirs, 1)

    # Test does not make assumptions about implementation of `VariableSource`. A typical
    # usage of it will only return the variables for specified keys. In this case,
    # `StubVariableSource` does not process the parameters at all.
    tree_utils.assert_equals(
        utils.restore_from_path(osp.join(self.tmp_dir.name, subdirs[0], "a")).params,
        parameters,
    )
    tree_utils.assert_equals(
        utils.restore_from_path(osp.join(self.tmp_dir.name, subdirs[0], "b")).params,
        parameters,
    )

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
    snap = snapshotter.Snapshotter(
        variable_source=test_utils.StubVariableSource([0.0]),
        snapshot_templates={
            "foo": utils.Snapshot(
                ctor=ctor,
                ctor_kwargs=ctor_kwargs,
                trace_kwargs=trace_kwargs,
            )
        },
        directory=self.tmp_dir.name,
    )
    snap.save()

    subdirs = sorted(file_utils.get_subdirs(self.tmp_dir.name))
    self.assertLen(subdirs, 1)
    snapshot = utils.restore_from_path(osp.join(self.tmp_dir.name, subdirs[0], "foo"))

    loaded_op = snapshot.ctor(**snapshot.ctor_kwargs)
    loaded_op = jax.jit(loaded_op.__call__)
    output = loaded_op(**snapshot.trace_kwargs)
    tree_utils.assert_equals(output, expected_output)


if __name__ == "__main__":
  absltest.main()
