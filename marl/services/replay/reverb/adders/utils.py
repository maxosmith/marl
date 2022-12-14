"""."""

from typing import Dict, Sequence

import jax.numpy as jnp
import numpy as np
import tree

from marl import _types
from marl.services.replay.reverb.adders import reverb_adder
from marl.utils import array_utils, tree_utils


def padding_mask(step: reverb_adder.Step) -> _types.Array:
    """Construct a binary mask with 0s for padded steps."""
    # This puts 1s on the last step and all padding.
    mask = jnp.cumsum(step.end_of_episode, axis=-1)
    mask = jnp.logical_not(mask)
    return mask


def final_step_like(step: reverb_adder.Step, next_observation: _types.Tree) -> reverb_adder.Step:
    """Return a list of steps with the final step zero-filled."""
    # Make zero-filled components so we can fill out the last step.
    zero_action, zero_reward, zero_discount, zero_extras = tree.map_structure(
        array_utils.zeros_like, (step.action, step.reward, step.discount, step.extras)
    )

    # Return a final step that only has next_observation.
    return reverb_adder.Step(
        observation=next_observation,
        action=zero_action,
        reward=zero_reward,
        discount=zero_discount,
        start_of_episode=False,
        extras=zero_extras,
    )


def calculate_priorities(
    priority_fns: reverb_adder.PriorityFnMapping, steps: Sequence[reverb_adder.Step]
) -> Dict[str, float]:
    """Helper used to calculate the priority of a sequence of steps.
    This converts the sequence of steps into a PriorityFnInput tuple where the
    components of each step (actions, observations, etc.) are stacked along the
    time dimension.

    Priorities are calculated for the sequence or transition that starts from
    step[0].next_observation. As a result, the stack of observations comes from
    steps[0:] whereas all other components (e.g. actions, rewards, discounts,
    extras) corresponds to steps[1:].

    NOTE: this means that all components other than the observation will be
    ignored from step[0]. This also means that step[0] is allowed to correspond to
    an "initial step" in which case the action, reward, discount, and extras are
    each None, which is handled properly by this function.

    Args:
      priority_fns: a mapping from table names to priority functions (i.e. a
        callable of type PriorityFn). The given function will be used to generate
        the priority (a float) for the given table.
      steps: a list of Step objects used to compute the priorities.

    Returns:
      A dictionary mapping from table names to the priority (a float) for the
      given collection of steps.
    """

    if any([priority_fn is not None for priority_fn in priority_fns.values()]):
        # Stack the steps and wrap them as PrioityFnInput.
        fn_input = reverb_adder.PriorityFnInput(*tree_utils.stack(steps))

    return {table: (priority_fn(fn_input) if priority_fn else 1.0) for table, priority_fn in priority_fns.items()}
