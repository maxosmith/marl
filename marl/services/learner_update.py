"""Learner update that allows the result directory to be changed."""
import functools
import itertools
from typing import Iterator, NamedTuple, Optional, Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import reverb
from absl import logging

from marl import specs, types, worlds
from marl.services import counter as counter_lib
from marl.services.arenas import loggers
from marl.services.replays.reverb import prefetch_client
from marl.utils import distributed_utils, spec_utils, stopwatch


class TrainingState(NamedTuple):
  """Training state consists of network parameters and optimiser state."""

  params: types.Params
  opt_state: optax.OptState
  rng_key: jax.random.PRNGKeyArray


class LearnerUpdate:
  """Service that trains a policy.

  Args:
    policy:
    optimizer:
    env_spec: Environment specification, used to (re)build policy parameters.
    random_key:
    data_iterator: Data iterator used to train the policy.
    devices: A list of all learning devices.
    logger:
    counter:
    step_key:
    frame_key:
    compute_policy_kl_convergence: Compute the KL difference between the current parameters
        and the previous parameters. This allows us to measure how much the policy has converged,
        but it is expensive to compute.
  """

  _PMAP_AXIS_NAME = "data"

  def __init__(
      self,
      policy: nn.Module,
      optimizer: optax.GradientTransformation,
      env_spec: specs.EnvironmentSpec,
      random_key: jax.random.PRNGKeyArray,
      data_iterator: prefetch_client.ReverbPrefetchClient,
      devices: Optional[Sequence[jax.Device]] = None,
      logger: Optional[loggers.BaseLogger] = None,
      counter: Optional[counter_lib.Counter] = None,
      step_key: str = "update_steps",
      frame_key: str = "update_frames",
      compute_policy_kl_convergence: bool = False,
  ):
    """Initialize a learner's update node."""
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
    self._policy = policy
    self._optimizer = optimizer
    self._env_spec = env_spec
    self._logger = logger
    self._counter = counter
    self._step_key = step_key
    self._frame_key = frame_key
    self._data_iterator = data_iterator
    self._compute_policy_kl_convergence = compute_policy_kl_convergence
    self._random_key = random_key
    self._stopwatch = stopwatch.Stopwatch(buffer_size=1_000)

    # @jax.jit
    def _update_step(
        params: types.Params,
        rng: jax.random.PRNGKey,
        opt_state: optax.OptState,
        sample: reverb.ReplaySample,
    ):
      """Computes and applies an update step on the learner's parameters."""
      rng_key, subkey = jax.random.split(rng)

      # Compute gradients.
      grad_fn = jax.value_and_grad(
          functools.partial(self._policy.apply, method="loss"),
          has_aux=True,
      )
      (_, metrics), gradients = grad_fn(params, sample)

      # Average gradients over pmap replicas before optimizer update.
      gradients = jax.lax.pmean(gradients, axis_name=LearnerUpdate._PMAP_AXIS_NAME)

      # Apply updates.
      updates, new_opt_state = optimizer.update(gradients, opt_state)
      new_params = optax.apply_updates(params, updates)

      # metrics.update({
      #     "param_norm": optax.global_norm(new_params),
      #     "param_updates_norm": optax.global_norm(updates),
      # })
      new_state = TrainingState(
          params=new_params, opt_state=new_opt_state, rng_key=rng_key
      )
      return new_state, metrics

    self._update_step = jax.pmap(
        _update_step, axis_name=LearnerUpdate._PMAP_AXIS_NAME, devices=self._devices
    )
    self._state = None
    self.reset_training_state()

  def run(self, num_steps: Optional[int] = None) -> None:
    """Run the update loop; typically an infinite loop which calls step.

    Args:
        num_steps: Number of learner update steps to perform.
    """
    logging.info("Running training for %s steps.", num_steps)
    step_iter = range(num_steps) if num_steps is not None else itertools.count()
    for _ in step_iter:
      self.step(self._data_iterator)

  def step(
      self,
      data_iter: Iterator[worlds.Trajectory | reverb.ReplaySample] | None = None,
  ):
    """Perform a single update step."""
    self._stopwatch.start("fetch")
    sample = next(data_iter) if data_iter else next(self._data_iterator)
    if isinstance(sample, reverb.ReplaySample):
      sample_data = sample.data
    else:
      sample_data = sample
    self._stopwatch.stop("fetch")

    self._stopwatch.start("update")
    self._state, metrics = self._update_step(
        params=self._state.params,
        rng=self._state.rng_key,
        opt_state=self._state.opt_state,
        sample=sample_data,
    )
    self._stopwatch.stop("update")

    # Take results from first replica.
    # NOTE: This measure will be a noisy estimate for the purposes
    #  of the logs as it does not pmean over all devices.
    metrics = distributed_utils.get_from_first_device(metrics)

    # Time metadata.
    times = self._stopwatch.get_splits(aggregate_fn=np.mean)
    for key, value in times.items():
      metrics[f"times/{key}"] = value

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

  def get_variables(self, names: Optional[Sequence[str]] = None) -> types.Params:
    """Retrieve the learner variables."""
    # Return first replica of parameters.
    if names is not None:
      raise NotImplementedError("Getting specific variable collections not supported.")
    del names
    return distributed_utils.get_from_first_device(
        [self._state.params],
        as_numpy=False,
    )[0]

  def save(self) -> TrainingState:
    """Retrieve the learner state to be saved."""
    return distributed_utils.get_from_first_device(self._state)

  def restore(self, state: TrainingState):
    """Restore the learner state."""
    self._state = distributed_utils.replicate_on_all_devices(state, self._local_devices)

  def reset_training_state(self):
    """Reset the learner's parameters and optimizer state."""
    self._random_key, subkey = jax.random.split(self._random_key)
    dummy_timestep = worlds.TimeStep(
        step_type=None,
        observation=spec_utils.zeros_like(self._env_spec.observation),
        reward=spec_utils.zeros_like(self._env_spec.reward),
    )
    dummy_agent_state = self._policy.initialize_carry(subkey, ())
    params = self._policy.init(subkey, dummy_agent_state, dummy_timestep)
    opt_state = self._optimizer.init(params)
    self._state = TrainingState(
        params=params, opt_state=opt_state, rng_key=self._random_key
    )
    self._state = distributed_utils.replicate_on_all_devices(
        self._state, self._local_devices
    )
