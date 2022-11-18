import chex
import jax.numpy as jnp

from marl import _types


def explained_variance(y: _types.Array, pred: _types.Array) -> _types.Array:
    """Computes the explained variance for a pair of labels and predictions.

    The formula used is:
        max(-1.0, 1.0 - (std(y - pred)^2 / std(y)^2))

    Args:
        y: The labels.
        pred: The predictions.

    Returns:
        The explained variance given a pair of labels and predictions.
    """
    chex.assert_rank([y, pred], [2, 2])
    chex.assert_type([y, pred], float)
    chex.assert_equal_shape([y, pred])

    y_var = jnp.var(y, axis=0)
    diff_var = jnp.var(y - pred, axis=0)
    # Model case in which y does not vary with explained variance of -1
    return jnp.where(y_var == 0.0, -1.0, jnp.maximum(-1.0, 1 - (diff_var / y_var))[0])


