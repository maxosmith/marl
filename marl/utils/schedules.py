"""Schedules for changing variables across time."""
import abc
import dataclasses

import jax.numpy as jnp


class Schedule(abc.ABC):
  """Interface for time based function."""

  @abc.abstractmethod
  def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
    """Computes the schedule value at this time.

    Args:
      t: Time.

    Returns:
      Value of the schedule at time `t`.
    """


@dataclasses.dataclass
class Constant(Schedule):
  """Constant value.

  Args:
    x: Constant value of the schedule.
  """

  x: float

  def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
    """Computes the schedule value at this time."""
    return jnp.asarray(self.x)


@dataclasses.dataclass
class Linear(Schedule):
  """Linear function.

  Args:
    x_initial: Value of the schedule at time 0.
    x_final: Value ofthe schedule at time `num_steps`.
    num_steps: Final value of the schedule.
  """

  x_initial: float
  x_final: float
  num_steps: int

  def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
    """Computes the schedule value at this time."""
    fraction = jnp.amin(jnp.asarray([t / float(self.num_steps), 1.0]))
    return self.x_initial + fraction * (self.x_final - self.x_initial)


@dataclasses.dataclass
class Step(Schedule):
  """Steps from one value to another after a time threshold.

  Args:
    x_final: Value of the schedule before `num_steps`.
    num_steps: Time when value should switch.
    x_initial: Value of the schedule after `num_steps`.
  """

  x_final: float
  num_steps: int
  x_initial: float = 0.0

  def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
    """Computes the schedule value at this time."""
    return self.x_final if t >= self.num_steps else self.x_initial


@dataclasses.dataclass
class RectifiedLinear(Schedule):
  """Rectified Linear function.

  Args:
    x_initial: Value of the schedule when the unit is not active
      (linear function not started).
    x_final: Value at the end of the schedule.
    num_steps_start: Time when to start the linear schedule.
    num_steps_end: Time when to end the linear schedule.
  """

  x_initial: float
  x_final: float
  num_steps_start: int
  num_steps_end: int

  def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
    """Computes the schedule value at this time."""
    if t < self.num_steps_start:
      return self.x_initial
    else:
      fraction = jnp.amin(
          jnp.asarray([
              (t - self.num_steps_start) / float(self.num_steps_end - self.num_steps_start),
              1.0,
          ])
      )
      return self.x_initial + fraction * (self.x_final - self.x_initial)
