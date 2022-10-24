"""

References:
 - https://github.com/deepmind/acme/blob/77fb814eba749946a3e31ac2cb70f5ec4c9bff3c/acme/agents/jax/impala/learning.py
"""
import itertools
from typing import NamedTuple, Optional, Sequence

import haiku as hk
import jax
import optax
import reverb
import tree
from absl import logging

from marl import _types, services
from marl.services import counter as counter_lib
from marl.utils import distributed_utils, loggers


class TrainingState(NamedTuple):
    """Training state consists of network parameters and optimiser state."""

    params: _types.Params
    opt_state: optax.OptState


class LearnerUpdate:
    """Service providing a learner's policy/step function."""

    _PMAP_AXIS_NAME = "data"

    def __init__(
        self,
        loss_fn: hk.Transformed,
        optimizer: optax.GradientTransformation,
        random_key: jax.random.KeyArray,
        data_iterator: Sequence[reverb.ReplaySample],
        logger: Optional[loggers.Logger] = None,
        counter: Optional[counter_lib.Counter] = None,
        step_key: str = "update_steps",
        frame_key: str = "update_frames",
    ):
        """Initialize a learner's update node.

        Args:
            agent:
            optimizer:
            reverb_client:
            random_key:
        """
        self._logger = logger
        self._counter = counter
        self._step_key = step_key
        self._frame_key = frame_key
        self._data_iterator = data_iterator
        self._random_key = random_key

        # @jax.jit
        def _update_step(
            params: _types.Params, rng: jax.random.PRNGKey, opt_state: optax.OptState, sample: reverb.ReplaySample
        ):
            # Compute gradients.
            grad_fn = jax.value_and_grad(loss_fn.apply, has_aux=True)
            (_, metrics), gradients = grad_fn(params, rng, sample)

            # Apply updates.
            updates, new_opt_state = optimizer.update(gradients, opt_state)
            new_params = optax.apply_updates(params, updates)

            metrics.update(
                {
                    "param_norm": optax.global_norm(new_params),
                    "param_updates_norm": optax.global_norm(updates),
                }
            )
            new_state = TrainingState(params=new_params, opt_state=new_opt_state)
            return new_state, metrics

        self._update_step = _update_step
        params = loss_fn.init(random_key)
        opt_state = optimizer.init(params)
        self._state = TrainingState(params=params, opt_state=opt_state)

    def run(self, num_steps: Optional[int] = None) -> None:
        """Run the update loop; typically an infinite loop which calls step."""
        iterator = range(num_steps) if num_steps is not None else itertools.count()

        for _ in iterator:
            self.step()

    def step(self):
        """Perform a single update step."""
        reverb_sample = next(self._data_iterator)
        # Remove device dimension that was added by the prefetch client.
        reverb_sample = tree.map_structure(lambda x: x[0], reverb_sample)

        self._state, metrics = self._update_step(
            params=self._state.params,
            rng=self._random_key,
            opt_state=self._state.opt_state,
            sample=reverb_sample.data,
        )

        if self._counter:
            counts = self._counter.increment(
                **{
                    self._step_key: 1,
                }
            )
            metrics[self._step_key] = counts[self._step_key]

        if self._logger:
            self._logger.write(metrics)

    def get_variables(self, names: Optional[Sequence[str]] = None) -> _types.Params:
        # Return first replica of parameters.
        del names
        return [self._state.params]

    def save(self) -> TrainingState:
        return self._state

    def restore(self, state: TrainingState):
        self._state = state

    def _get_step(self) -> int:
        """Get the current global step count."""
        counts = self._counter.get_counts()
        return counts.get(self._step_key, 0), counts.get(self._frame_key, 0)
