"""

References:
 - https://github.com/deepmind/acme/blob/77fb814eba749946a3e31ac2cb70f5ec4c9bff3c/acme/agents/jax/impala/learning.py
"""
import itertools
from typing import NamedTuple, Optional, Sequence

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import reverb
import tree
from absl import logging

from marl import _types, services
from marl.services import counter as counter_lib
from marl.services.replay.reverb.adders import utils as reverb_utils
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
        devices: Optional[Sequence[jax.xla.Device]] = None,
        logger: Optional[loggers.Logger] = None,
        counter: Optional[counter_lib.Counter] = None,
        step_key: str = "update_steps",
        frame_key: str = "update_frames",
        compute_policy_kl_convergence: bool = True,
    ):
        """Initialize a learner's update node.

        Args:
            agent:
            optimizer:
            reverb_client:
            random_key:
            devices: A list of all learning devices.
            compute_policy_kl_convergence: Compute the KL difference between the current parameters
                and the previous parameters. This allows us to measure how much the policy has converged,
                but it is expensive to compute.
        """
        local_devices = jax.local_devices()
        process_id = jax.process_index()
        self._devices = devices or local_devices  # Requiring PMAP.
        self._local_devices = [d for d in self._devices if d in local_devices]
        logging.info(
            "Initializing learner process ID %s. Passed devices: %s; local devices: %s.",
            process_id,
            devices,
            local_devices,
        )
        self._logger = logger
        self._counter = counter
        self._step_key = step_key
        self._frame_key = frame_key
        self._data_iterator = data_iterator
        self._compute_policy_kl_convergence = compute_policy_kl_convergence

        @jax.jit
        def _update_step(
            params: _types.Params,
            rng: jax.random.PRNGKey,
            opt_state: optax.OptState,
            sample: reverb.ReplaySample,
        ):
            """Computes and applies an update step on the learner's parameters."""
            # Compute gradients.
            grad_fn = jax.value_and_grad(loss_fn.apply, has_aux=True)
            (_, metrics), gradients = grad_fn(params, rng, sample)

            # Average gradients over pmap replicas before optimizer update.
            gradients = jax.lax.pmean(gradients, axis_name=LearnerUpdate._PMAP_AXIS_NAME)

            # Apply updates.
            updates, new_opt_state = optimizer.update(gradients, opt_state)
            new_params = optax.apply_updates(params, updates)

            metrics.update(
                {
                    "param_norm": optax.global_norm(new_params),
                    "param_updates_norm": optax.global_norm(updates),
                }
            )

            # If metrics include policy delete it, this is used for measuring policy
            # convergence in `_policy_convergence`.
            if "policy" in metrics:
                del metrics["policy"]

            new_state = TrainingState(params=new_params, opt_state=new_opt_state)
            return new_state, metrics

        @jax.jit
        def _policy_convergence(
            params: _types.Params,
            prev_params: _types.Params,
            rng: jax.random.PRNGKey,
            sample: reverb.ReplaySample,
        ):
            _, metrics = loss_fn.apply(params, rng, sample)
            _, prev_metrics = loss_fn.apply(prev_params, rng, sample)

            policy = distrax.Categorical(probs=metrics["policy"])
            prev_policy = distrax.Categorical(probs=prev_metrics["policy"])

            return jnp.mean(policy.kl_divergence(prev_policy))

        self._update_step = jax.pmap(_update_step, axis_name=LearnerUpdate._PMAP_AXIS_NAME, devices=self._devices)
        self._policy_convergence = jax.pmap(
            _policy_convergence, axis_name=LearnerUpdate._PMAP_AXIS_NAME, devices=self._devices
        )

        params = loss_fn.init(random_key)
        opt_state = optimizer.init(params)
        self._state = TrainingState(params=params, opt_state=opt_state)
        self._state = distributed_utils.replicate_on_all_devices(self._state, self._local_devices)
        self._random_key = distributed_utils.replicate_on_all_devices(random_key, self._local_devices)

    def run(self, num_steps: Optional[int] = None) -> None:
        """Run the update loop; typically an infinite loop which calls step."""
        iterator = range(num_steps) if num_steps is not None else itertools.count()

        logging.info("Running training for %s steps.", num_steps)

        for _ in iterator:
            self.step()

    def step(self):
        """Perform a single update step."""
        reverb_sample = next(self._data_iterator)

        split_key = jax.random.split(self._random_key)
        self._random_key, subkey = split_key[:, 0], split_key[:, 1]  # Ignore device axis.

        prev_params = self._state.params
        self._state, metrics = self._update_step(
            params=self._state.params,
            rng=subkey,
            opt_state=self._state.opt_state,
            sample=reverb_sample.data,
        )

        # Take results from first replica.
        # NOTE: This measure will be a noisy estimate for the purposes
        #  of the logs as it does not pmean over all devices.
        metrics = distributed_utils.get_from_first_device(metrics)

        if self._compute_policy_kl_convergence:
            policy_kl_convergence = self._policy_convergence(
                params=self._state.params,
                prev_params=prev_params,
                rng=subkey,
                sample=reverb_sample.data,
            )
            policy_kl_convergence = distributed_utils.get_from_first_device(policy_kl_convergence)
            metrics["policy_kl_convergence"] = policy_kl_convergence

        # Reverb metadata.
        metrics["replay/times_sampled"] = jnp.mean(reverb_sample.info.times_sampled)
        metrics["replay/priority"] = jnp.mean(reverb_sample.info.priority)
        metrics["replay/probability"] = jnp.mean(reverb_sample.info.probability)
        metrics["replay/table_size"] = jnp.mean(reverb_sample.info.table_size)

        if self._counter:
            counts = self._counter.increment(
                **{
                    self._step_key: 1,
                    self._frame_key: metrics["batch_size"].tolist(),
                }
            )
            metrics[self._step_key] = counts[self._step_key]
            metrics[self._frame_key] = counts[self._frame_key]

        if self._logger:
            self._logger.write(metrics)

    def get_variables(self, names: Optional[Sequence[str]] = None) -> _types.Params:
        # Return first replica of parameters.
        del names
        return distributed_utils.get_from_first_device([self._state.params], as_numpy=False)

    def save(self) -> TrainingState:
        return distributed_utils.get_from_first_device(self._state)

    def restore(self, state: TrainingState):
        self._state = distributed_utils.replicate_on_all_devices(state, self._local_devices)

    def _get_step(self) -> int:
        """Get the current global step count."""
        counts = self._counter.get_counts()
        return counts.get(self._step_key, 0), counts.get(self._frame_key, 0)
