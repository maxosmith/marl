import os.path as osp
import tempfile

import haiku as hk
import jax
from absl.testing import absltest, parameterized

from marl import _types, games, worlds
from marl.services import snapshotter
from marl.utils import mocks, spec_utils, tree_utils


def model_factory(num_powers: int):
    def _model(params: _types.Params, x: _types.Array) -> _types.Array:
        return params["W"] ** num_powers + x**num_powers

    return _model


class SnapshotterTest(parameterized.TestCase):
    """Test cases for `LearnerPolicy`."""

    def test_learner_policy(self):
        snapshot = snapshotter.Snapshot(
            ctor=model_factory,
            ctor_kwargs={"num_powers": 2},
            trace_kwargs={"x": 0},
            params={"W": 1},
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshotter.save_to_path(tmp_dir, snapshot)
            restored = snapshotter.restore_from_path(tmp_dir)

        tree_utils.assert_equals(snapshot, restored)


if __name__ == "__main__":
    absltest.main()
