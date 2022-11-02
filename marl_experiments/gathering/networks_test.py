import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from marl_experiments.gathering import networks


class NetworksTest(parameterized.TestCase):
    @hk.testing.transform_and_run
    def test_world_model_prediction(self):
        module = networks.WorldStateLinearPredictionHead(state_shape=(21, 19, 4))
        print(module(jnp.zeros([2, 200])).shape)

    @hk.testing.transform_and_run
    def test_world_model_encoder(self):
        module = networks.WorldStateLinearEncoder(state_shape=(21, 19, 4), num_actions=8)
        print(module(jnp.zeros([2, 21, 19, 4]), {0: np.ones([2]), 1: np.ones([2])}).shape)

    @hk.testing.transform_and_run
    def test_world_model_encoder_prediction(self):
        enc = networks.WorldStateLinearEncoder(state_shape=(21, 19, 4), num_actions=8)
        dec = networks.WorldStateLinearPredictionHead(state_shape=(21, 19, 4))
        print(dec(enc(jnp.zeros([2, 21, 19, 4]), {0: np.ones([2]), 1:np.ones([2])})).shape)


if __name__ == "__main__":
    absltest.main()
