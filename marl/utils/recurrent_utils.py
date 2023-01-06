"""JAX utility functions for recurrent networks.."""
from typing import Any, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp


class ShapePreservingCore(hk.RNNCore):
    """."""

    def __init__(self, core: hk.RNNCore, name: Optional[str] = None):
        """Initializes a ShapePreservingCore."""
        super().__init__(name=name)
        self._core = core

    def __call__(self, inputs, prev_state) -> Tuple[Any, Any]:
        """."""
        output, next_state = self._core(inputs, prev_state)
        # Project the output embedding to be the same shape as the input embedding.
        # This is useful for auto-regressive sampling.
        if output.shape[-1] != inputs.shape[-1]:
            output = hk.Linear(inputs.shape[-1])(output)
        return output, next_state

    def initial_state(self, batch_size: Optional[int]):
        """."""
        return self._core.initial_state(batch_size)


def static_autoregressive_unroll(core, input_sequence, initial_state, time_major=True, epsilon: float = 0.0):
    """

    TODO(maxsmith): docstring https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/recurrent.py


    Args:
        epsilon: Portion of truth given to the network (via scheduled sampling).
    """
    output_sequence = []
    time_axis = 0 if time_major else 1
    num_steps = jax.tree_util.tree_leaves(input_sequence)[0].shape[time_axis]
    state = initial_state

    # First input is pulled from the input sequence.
    if time_major:
        inputs = jax.tree_util.tree_map(lambda x: x[0], input_sequence)
    else:
        inputs = jax.tree_util.tree_map(lambda x: x[:, 0], input_sequence)

    # The remainder of the unroll is autoregressive.
    for t in range(num_steps):
        outputs, state = core(inputs, state)
        output_sequence.append(outputs)

        if time_major:
            truth = jax.tree_util.tree_map(lambda x, _t=t: x[_t], input_sequence)
        else:
            truth = jax.tree_util.tree_map(lambda x, _t=t: x[:, _t], input_sequence)

        # Epsilon % of the time, feed the true input into the model.
        truth_mask = jax.random.uniform(hk.next_rng_key(), shape=(inputs.shape[0],)) < epsilon
        inputs = jnp.where(truth_mask[..., None], truth, outputs)

    # Stack outputs along the time axis.
    output_sequence = jax.tree_util.tree_map(
        lambda *args: jnp.stack(args, axis=time_axis),
        *output_sequence,
    )
    return output_sequence, state
