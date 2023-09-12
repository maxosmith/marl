"""Worlds that agent(s) that agents interact with.

These are largely forked from `dm_env` and have the unused
`discount` field deprecated. Specifically, the `Environment`
and `TimeStep` objects.

https://github.com/deepmind/dm_env/blob/master/dm_env/_environment.py
"""
import abc
import enum
from typing import Any, Mapping, NamedTuple, Tuple, Union

import tensorflow as tf

from marl import specs, types


class TimeStep(NamedTuple):
  """Returned with every call to `step` and `reset` on an environment.

  A `TimeStep` contains the data emitted by an environment at each step of
  interaction. A `TimeStep` holds a `step_type`, an `observation` (typically a
  NumPy array or a dict or list of arrays), and an associated `reward`.

  The first `TimeStep` in a sequence will have `StepType.FIRST`. The final
  `TimeStep` will have `StepType.LAST`. All other `TimeStep`s in a sequence will
  have `StepType.MID.

  Attributes:
      step_type: A `StepType` enum value.
      reward:  A scalar, NumPy array, nested dict, list or tuple of rewards; or
          `None` if `step_type` is `StepType.FIRST`, i.e. at the start of a
          sequence.
      observation: A NumPy array, or a nested dict, list or tuple of arrays.
          Scalar values that can be cast to NumPy arrays (e.g. Python floats) are
          also valid in place of a scalar array.
  """

  step_type: Any
  reward: Any
  observation: Any

  def first(self) -> bool:
    return self.step_type == StepType.FIRST

  def mid(self) -> bool:
    return self.step_type == StepType.MID

  def last(self) -> bool:
    return self.step_type == StepType.LAST


PlayerIDToTimestep = Mapping[types.PlayerID, TimeStep]


class Trajectory(NamedTuple):
  """Sequence of TimeStep(s)."""

  observation: types.Tree
  action: types.Tree
  reward: types.Tree
  start_of_episode: Union[bool, types.Array, tf.Tensor, Tuple[()]]
  end_of_episode: Union[bool, types.Array, tf.Tensor, Tuple[()]]
  extras: types.Tree = ()


class StepType(enum.IntEnum):
  """Defines the status of a `TimeStep` within a sequence."""

  # Denotes the first `TimeStep` in a sequence.
  FIRST = 0
  # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
  MID = 1
  # Denotes the last `TimeStep` in a sequence.
  LAST = 2

  def first(self) -> bool:
    return self is StepType.FIRST

  def mid(self) -> bool:
    return self is StepType.MID

  def last(self) -> bool:
    return self is StepType.LAST


class Environment(metaclass=abc.ABCMeta):
  """Abstract base class for Python RL environments.

  Observations and valid actions are described with `Array` specs, defined in
  the `specs` module.
  """

  @abc.abstractmethod
  def reset(self) -> TimeStep:
    """Starts a new sequence and returns the first `TimeStep` of this sequence.

    Returns:
        A `TimeStep` namedtuple containing:
            step_type: A `StepType` of `FIRST`.
            reward: `None`, indicating the reward is undefined.
            observation: A NumPy array, or a nested dict, list or tuple of arrays.
            Scalar values that can be cast to NumPy arrays (e.g. Python floats)
            are also valid in place of a scalar array. Must conform to the
            specification returned by `observation_spec()`.
    """

  @abc.abstractmethod
  def step(self, action: types.Action) -> TimeStep:
    """Updates the environment according to the action and returns a `TimeStep`.

    If the environment returned a `TimeStep` with `StepType.LAST` at the
    previous step, this call to `step` will start a new sequence and `action`
    will be ignored.

    This method will also start a new sequence if called after the environment
    has been constructed and `reset` has not been called. Again, in this case
    `action` will be ignored.

    Args:
        action: A NumPy array, or a nested dict, list or tuple of arrays
            corresponding to `action_spec()`.

    Returns:
        A `TimeStep` namedtuple containing:
            step_type: A `StepType` value.
            reward: Reward at this timestep, or None if step_type is
            `StepType.FIRST`. Must conform to the specification returned by
            `reward_spec()`.
            observation: A NumPy array, or a nested dict, list or tuple of arrays.
            Scalar values that can be cast to NumPy arrays (e.g. Python floats)
            are also valid in place of a scalar array. Must conform to the
            specification returned by `observation_spec()`.
    """

  def spec(self) -> specs.EnvironmentSpec:
    """Describes the environment.

    Returns:
        An `EnvironmentSpec` describing the reward, observation, and action spaces.
    """
    return specs.EnvironmentSpec(
        observation=self.observation_spec(),
        action=self.action_spec(),
        reward=self.reward_spec(),
    )

  def reward_spec(self) -> specs.TreeSpec:
    """Describes the reward returned by the environment.

    Returns:
        An `Array` spec, or a nested dict, list or tuple of `Array` specs.
    """
    return specs.ArraySpec(shape=(), dtype=float, name="reward")

  @abc.abstractmethod
  def observation_spec(self) -> specs.TreeSpec:
    """Defines the observations provided by the environment.

    Returns:
        An `Array` spec, or a nested dict, list or tuple of `Array` specs.
    """

  @abc.abstractmethod
  def action_spec(self) -> specs.TreeSpec:
    """Defines the actions that should be provided to `step`.

    Returns:
        An `Array` spec, or a nested dict, list or tuple of `Array` specs.
    """

  def close(self):
    """Frees any resources used by the environment.

    Implement this method for an environment backed by an external process.

    This method can be used directly
        ```python
        env = Env(...)
        # Use env.
        env.close()
        ```

    or via a context manager
        ```python
        with Env(...) as env:
        # Use env.
        ```
    """
    pass

  def __enter__(self):
    """Allows the environment to be used in a with-statement context."""
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    """Allows the environment to be used in a with-statement context."""
    del exc_type, exc_value, traceback  # Unused.
    self.close()


class Game(metaclass=abc.ABCMeta):
  """Abstract base class for Python RL environments.

  Observations and valid actions are described with `Array` specs, defined in
  the `specs` module.
  """

  @abc.abstractmethod
  def reset(self) -> PlayerIDToTimestep:
    """Starts a new sequence and returns the first `TimeStep` of this sequence.

    Returns:
        A `TimeStep` namedtuple containing:
            step_type: A `StepType` of `FIRST`.
            reward: `None`, indicating the reward is undefined.
            observation: A NumPy array, or a nested dict, list or tuple of arrays.
            Scalar values that can be cast to NumPy arrays (e.g. Python floats)
            are also valid in place of a scalar array. Must conform to the
            specification returned by `observation_spec()`.
    """

  @abc.abstractmethod
  def step(self, actions: types.PlayerIDToAction) -> PlayerIDToTimestep:
    """Updates the environment according to the action and returns a `TimeStep`.

    If the environment returned a `TimeStep` with `StepType.LAST` at the
    previous step, this call to `step` will start a new sequence and `action`
    will be ignored.

    This method will also start a new sequence if called after the environment
    has been constructed and `reset` has not been called. Again, in this case
    `action` will be ignored.

    Args:
        action: A NumPy array, or a nested dict, list or tuple of arrays
            corresponding to `action_spec()`.
    Returns:
        A `TimeStep` namedtuple containing:
            step_type: A `StepType` value.
            reward: Reward at this timestep, or None if step_type is
            `StepType.FIRST`. Must conform to the specification returned by
            `reward_spec()`.
            observation: A NumPy array, or a nested dict, list or tuple of arrays.
            Scalar values that can be cast to NumPy arrays (e.g. Python floats)
            are also valid in place of a scalar array. Must conform to the
            specification returned by `observation_spec()`.
    """

  def spec(self) -> specs.GameSpec:
    """Describes the game interface for each player.

    Returns:
        An `EnvironmentSpec` for each player.
    """
    reward_specs = self.reward_specs()
    observation_specs = self.observation_specs()
    action_specs = self.action_specs()
    return {
        id: specs.EnvironmentSpec(
            observation=observation_specs[id],
            action=action_specs[id],
            reward=reward_specs[id],
        )
        for id in reward_specs.keys()
    }

  @abc.abstractmethod
  def reward_specs(self) -> specs.PlayerIDToSpec:
    """Describes the reward returned by the game to each player.

    Returns:
        A specification of the reward space for each player.
    """

  @abc.abstractmethod
  def observation_specs(self) -> specs.PlayerIDToSpec:
    """Defines the observations provided by the game to each player.

    Returns:
        A specification of the observation space for each player.
    """

  @abc.abstractmethod
  def action_specs(self) -> specs.PlayerIDToSpec:
    """Defines the actions that should be provided to `step` by each player.

    Returns:
        A specification of the action space for each player.
    """

  def close(self):
    """Frees any resources used by the game.

    Implement this method for an game backed by an external process.

    This method can be used directly
        ```python
        env = Env(...)
        # Use env.
        env.close()
        ```

    or via a context manager
        ```python
        with Env(...) as env:
        # Use env.
        ```
    """
    pass

  def __enter__(self):
    """Allows the game to be used in a with-statement context."""
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    """Allows the game to be used in a with-statement context."""
    del exc_type, exc_value, traceback  # Unused.
    self.close()
