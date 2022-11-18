"""Performs behavioural cloning for the RPS bots."""
import itertools
from typing import Any, Iterator, NamedTuple, Optional, Sequence

import haiku as hk
import jax
import optax
import reverb
import tree
from absl import logging

from marl import _types, services, worlds
from marl.services import counter as counter_lib
from marl.utils import distributed_utils, loggers


class TrainingState(NamedTuple):
    """Training state consists of network parameters and optimiser state."""

    params: _types.Params
    opt_state: optax.OptState


class Learner:
    """Service providing a learner's policy/step function."""

    def __init__(
        self,
        loss_fn: hk.Transformed,
        optimizer: optax.GradientTransformation,
        key_sequence: hk.PRNGSequence,
        data_iterator: Iterator[Any],
        logger: Optional[loggers.Logger] = None,
        counter: Optional[counter_lib.Counter] = None,
        step_key: str = "learner_steps",
        frame_key: str = "learner_frames",
    ):
        """Initialize a learner's update node.

        Args:
            agent:
            optimizer:
            reverb_client:
        """
        self._logger = logger
        self._counter = counter
        self._step_key = step_key
        self._frame_key = frame_key
        self._data_iterator = data_iterator
        self._key_sequence = key_sequence

        @jax.jit
        def _update_step(
            params: _types.Params,
            rng: jax.random.PRNGKey,
            opt_state: optax.OptState,
            sample: Any,
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
        # TODO(maxsmith): Refactor so that parameter initialization does not require a real batch.
        params = loss_fn.init(next(self._key_sequence), self._preprocess_data(next(self._data_iterator)))
        opt_state = optimizer.init(params)
        self._state = TrainingState(params=params, opt_state=opt_state)

    def run(self, num_steps: Optional[int] = None) -> None:
        """Run the update loop; typically an infinite loop which calls step."""
        iterator = range(num_steps) if num_steps is not None else itertools.count()

        for _ in iterator:
            self.step()

    def step(self):
        """Perform a single update step."""
        batch = next(self._data_iterator)
        batch = self._preprocess_data(batch)

        self._state, metrics = self._update_step(
            params=self._state.params,
            rng=next(self._key_sequence),
            opt_state=self._state.opt_state,
            sample=batch,
        )

        if self._counter:
            counts = {self._step_key: 1}
            counts = self._counter.increment(**counts)
            metrics[self._step_key] = counts[self._step_key]

        if self._logger:
            self._logger.write(metrics)

    def _preprocess_data(self, batch):
        """Preprocess the data to match the expected API for the update network."""
        observation = batch["observations"]  # [B, T, 6].
        demonstration = batch["actions"][..., 0]  # [B, T].
        self_id = batch["bot_ids"][..., 0]  # [B, T].
        opp_id = batch["bot_ids"][..., 1]  # [B, T].
        padding_mask = batch["padding_mask"]  # [B, T].

        # Convert the batch to match time Timetep interface. Only the observation
        # field is used by the TimestepEncoder, so fill the rest with dummy values.
        timestep = worlds.TimeStep(
            step_type=0,
            observation={
                "info_state": observation,
                "self_id": self_id,
                "opp_id": opp_id,
                "demonstration": demonstration,
                "padding_mask": padding_mask,
            },
            reward=0,
        )
        return timestep

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
