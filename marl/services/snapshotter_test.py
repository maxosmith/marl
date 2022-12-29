"""Test suite for the `Snapshotter` service."""
import tempfile

from absl.testing import absltest, parameterized

from marl import _types
from marl.services import snapshotter
from marl.utils import tree_utils


def model_factory(num_powers: int):
    """Factory for dummy models."""

    def _model(params: _types.Params, x: _types.Array) -> _types.Array:
        """Simple model."""
        return params["W"] ** num_powers + x**num_powers

    return _model


class SnapshotterTest(parameterized.TestCase):
    """Test cases for `Snapshotter`."""

    @parameterized.named_parameters(
        [
            ("save_bundled", snapshotter.save_to_path_bundled),
            ("save_unbundled", snapshotter.save_to_path),
        ]
    )
    def test_save_restore(self, save_fn):
        """Test basic saving/restoring of a snapshot."""
        snapshot = snapshotter.Snapshot(
            ctor=model_factory,
            ctor_kwargs={"num_powers": 2},
            trace_kwargs={"x": 0},
            params={"W": 1},
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_fn(tmp_dir, snapshot)
            restored = snapshotter.restore_from_path(tmp_dir)

        tree_utils.assert_equals(snapshot, restored)


if __name__ == "__main__":
    absltest.main()
