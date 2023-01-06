import abc
import dataclasses

import jax.numpy as jnp


class Schedule(abc.ABC):
    @abc.abstractmethod
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Computes the schedule value at this time."""


@dataclasses.dataclass
class Constant(Schedule):

    x: float

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(self.x)


@dataclasses.dataclass
class Linear(Schedule):

    x_initial: float
    x_final: float
    num_steps: int

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        fraction = jnp.amin(jnp.asarray([t / float(self.num_steps), 1.0]))
        return self.x_initial + fraction * (self.x_final - self.x_initial)


@dataclasses.dataclass
class Step(Schedule):

    x_final: float
    num_steps: int
    x_initial: float = 0.0

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        return self.x_final if t > self.num_steps else self.x_initial


@dataclasses.dataclass
class RectifiedLinear(Schedule):

    x_initial: float
    x_final: float
    num_steps_start: int
    num_steps_end: int

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        if t < self.num_steps_start:
            return self.x_initial
        else:
            fraction = jnp.amin(
                jnp.asarray([(t - self.num_steps_start) / float(self.num_steps_end - self.num_steps_start), 1.0])
            )
            return self.x_initial + fraction * (self.x_final - self.x_initial)
