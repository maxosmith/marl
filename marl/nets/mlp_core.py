"""MLP conforming to the RNNCore interface."""
from typing import Callable, Iterable, NamedTuple, Optional

import haiku as hk
import jax
import jax.numpy as jnp
import tree

from marl.utils import tree_utils


class MLPCoreState(NamedTuple):
    """A MLP core state consists of a dummy value."""

    dummy: jnp.ndarray


class MLPCore(hk.RNNCore):
    """A multi-layer perceptron with a recurrent interface."""

    def __init__(
        self,
        output_sizes: Iterable[int],
        activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
        activate_final: bool = False,
        name: Optional[str] = None,
    ):
        """Initialies and instance of `MLPCore`."""
        super().__init__(name=name)
        self._output_sizes = output_sizes
        self._activation = activation
        self._activate_final = activate_final

    def __call__(self, inputs: jnp.ndarray, prev_state: MLPCoreState) -> jnp.ndarray:
        """Connects the module to some inputs."""
        output = inputs
        for output_size in self._output_sizes[:-1]:
            output = hk.Linear(output_size=output_size)(output)
            output = self._activation(output)
        output = hk.Linear(output_size=self._output_sizes[-1])(output)
        if self._activate_final:
            output = self._activation(output)
        return output, prev_state

    def initial_state(self, batch_size: Optional[int]) -> MLPCoreState:
        state = MLPCoreState(dummy=jnp.zeros([1]))
        if batch_size is not None:
            state = tree_utils.add_batch(state, batch_size)
        return state
