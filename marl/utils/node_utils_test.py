"""Test for `node_utils`."""
import dataclasses
import threading
import time

import launchpad as lp
from absl.testing import absltest, parameterized
from launchpad.launch.test_multi_threading import address_builder

from marl.utils import node_utils


@dataclasses.dataclass
class _FakeService:
  """Terminates once it receives the first ping() call."""

  x: int
  y: float
  z: str = "default"

  def get_state(self):
    """Get the state of the service, since only methods are exposed."""
    return [self.x, self.y, self.z]

  def run(self):
    """Need to keep service alive since it's not managed by a `lp.Program`."""
    while True:
      time.sleep(1)


class BuildCourierNodeTest(parameterized.TestCase):
  """Test suite for `node_utils.build_courier_node`."""

  @parameterized.parameters(True, False)
  def test_disable_run(self, disable_run: bool):
    """Tests disabling run."""

    @node_utils.build_courier_node(disable_run=disable_run)
    def _build_service() -> _FakeService:
      """Builds a fake service for testing."""
      return _FakeService(x=42, y=3.14)

    service: lp.CourierNode = _build_service()
    # pylint: disable=protected-access
    self.assertEqual(service._should_run, not disable_run)
    # pylint: enable=protected-access

  @parameterized.parameters(
      dict(x=42, y=3.14, z="not_default"),
      dict(x=1, y=3, z="other"),
  )
  def test_args_kwargs(self, x: int, y: float, z: str):
    """Tests args/kwargs correctly passed."""

    @node_utils.build_courier_node
    def _build_service(x_: int, y_: float, z_: str) -> _FakeService:
      """Builds a fake service for testing."""
      return _FakeService(x_, y=y_, z=z_)

    node: lp.CourierNode = _build_service(x, y, z)
    address_builder.bind_addresses([node])  # Bind all addresses for testing.
    threading.Thread(target=node.run, daemon=True).start()
    client: _FakeService = node.create_handle().dereference()
    self.assertListEqual(client.get_state(), [x, y, z])


if __name__ == "__main__":
  absltest.main()
