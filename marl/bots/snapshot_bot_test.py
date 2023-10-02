"""Test for `snapshot_bot`."""
import pathlib
import tempfile

import jax
from absl.testing import absltest, parameterized

from marl.bots import snapshot_bot, test_utils
from marl.services import snapshotter


class SnapshotBotTest(parameterized.TestCase):
  """Test suite for `SnapshotBot`."""

  def test_step(self):
    """Tests step method."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      snapshot_path = pathlib.Path(tmp_dir) / "snapshot"

      params = {"params": {"mean": 2}}
      snapshot = snapshotter.Snapshot(
          ctor=test_utils.TestPolicyLinear,
          ctor_kwargs={},
          params=params,
      )
      snapshotter.save_to_path(snapshot_path, snapshot)
      bot = snapshot_bot.SnapshotBot(path=snapshot_path, rng_key=jax.random.PRNGKey(42))
      self.assertEqual(bot.params, params)
      state = bot.episode_reset(test_utils.FAKE_TIMESTEP)
      self.assertEqual(state, 1)
      action, state = bot.step(state, test_utils.FAKE_TIMESTEP)
      self.assertEqual(state, 2)
      self.assertEqual(action, 3)


if __name__ == "__main__":
  absltest.main()
