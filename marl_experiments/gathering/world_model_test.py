import haiku as hk
import numpy as np
from absl.testing import absltest, parameterized

from marl_experiments.gathering import networks, world_model


class WorldModelTest(parameterized.TestCase):
    @hk.testing.transform_and_run
    def test_loss(self):
        model = world_model.WorldModel()


if __name__ == "__main__":
    absltest.main()
