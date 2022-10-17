"""Worlds that agent(s) that agents interact with.

These are largely forked from `dm_env` and have the unused
`discount` field deprecated. Specifically, the `Environment`
and `TimeStep` objects.

https://github.com/deepmind/dm_env/blob/master/dm_env/_environment.py
"""
import enum
from typing import NamedTuple, Any, Mapping, Iterable, Union
import tensorflow as tf

import abc
from marl import _types
from dm_env import specs as dm_env_specs

ArraySpec = dm_env_specs.Array
BoundedArraySpec = dm_env_specs.BoundedArray
DiscreteArraySpec = dm_env_specs.DiscreteArray

TreeSpec = Union[
    ArraySpec,
    BoundedArraySpec,
    DiscreteArraySpec,
    Iterable["TreeSpec"],
    Mapping[Any, "TreeSpec"],
]

TreeTFSpec = Union[
    tf.TensorSpec,
    Iterable["TreeTFSpec"],
    Mapping[Any, "TreeTFSpec"],
]


class EnvironmentSpec(NamedTuple):
    """All domain specifications for an environment."""

    observation: TreeSpec
    action: TreeSpec
    reward: TreeSpec


GameSpec = Mapping[_types.PlayerID, EnvironmentSpec]

# Game specs.
PlayerIDToSpec = Mapping[_types.PlayerID, TreeSpec]
PlayerIDToEnvSpec = Mapping[_types.PlayerID, EnvironmentSpec]


class TimeStep(NamedTuple):
    """Returned with every call to `step` and `reset` on an environment.

    A `TimeStep` contains the data emitted by an environment at each step of
    interaction. A `TimeStep` holds a `step_type`, an `observation` (typically a
    NumPy array or a dict or list of arrays), and an associated `reward` and
    `discount`.

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


PlayerIDToTimestep = Mapping[_types.PlayerID, TimeStep]


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
                discount: `None`, indicating the discount is undefined.
                observation: A NumPy array, or a nested dict, list or tuple of arrays.
                Scalar values that can be cast to NumPy arrays (e.g. Python floats)
                are also valid in place of a scalar array. Must conform to the
                specification returned by `observation_spec()`.
        """

    @abc.abstractmethod
    def step(self, action: _types.Action) -> TimeStep:
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
                discount: A discount in the range [0, 1], or None if step_type is
                `StepType.FIRST`. Must conform to the specification returned by
                `discount_spec()`.
                observation: A NumPy array, or a nested dict, list or tuple of arrays.
                Scalar values that can be cast to NumPy arrays (e.g. Python floats)
                are also valid in place of a scalar array. Must conform to the
                specification returned by `observation_spec()`.
        """

    def spec(self) -> EnvironmentSpec:
        """Describes the environment.

        Returns:
            An `EnvironmentSpec` describing the reward, observation, and action spaces.
        """
        return EnvironmentSpec(
            observation=self.observation_spec(),
            action=self.action_spec(),
            reward=self.reward_spec(),
        )

    def reward_spec(self) -> TreeSpec:
        """Describes the reward returned by the environment.

        Returns:
            An `Array` spec, or a nested dict, list or tuple of `Array` specs.
        """
        return ArraySpec(shape=(), dtype=float, name="reward")

    @abc.abstractmethod
    def observation_spec(self) -> TreeSpec:
        """Defines the observations provided by the environment.

        Returns:
            An `Array` spec, or a nested dict, list or tuple of `Array` specs.
        """

    @abc.abstractmethod
    def action_spec(self) -> TreeSpec:
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
                discount: `None`, indicating the discount is undefined.
                observation: A NumPy array, or a nested dict, list or tuple of arrays.
                Scalar values that can be cast to NumPy arrays (e.g. Python floats)
                are also valid in place of a scalar array. Must conform to the
                specification returned by `observation_spec()`.
        """

    @abc.abstractmethod
    def step(self, actions: _types.PlayerIDToAction) -> PlayerIDToTimestep:
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
                discount: A discount in the range [0, 1], or None if step_type is
                `StepType.FIRST`. Must conform to the specification returned by
                `discount_spec()`.
                observation: A NumPy array, or a nested dict, list or tuple of arrays.
                Scalar values that can be cast to NumPy arrays (e.g. Python floats)
                are also valid in place of a scalar array. Must conform to the
                specification returned by `observation_spec()`.
        """

    def spec(self) -> GameSpec:
        """Describes the game interface for each player.

        Returns:
            An `EnvironmentSpec` for each player.
        """
        reward_specs = self.reward_specs()
        observation_specs = self.observation_specs()
        action_specs = self.action_specs()
        return {
            id: EnvironmentSpec(observation=observation_specs[id], action=action_specs[id], reward=reward_specs[id])
            for id in reward_specs.keys()
        }

    @abc.abstractmethod
    def reward_specs(self) -> PlayerIDToSpec:
        """Describes the reward returned by the game to each player.

        Returns:
            A specification of the reward space for each player.
        """

    @abc.abstractmethod
    def observation_specs(self) -> PlayerIDToSpec:
        """Defines the observations provided by the game to each player.

        Returns:
            A specification of the observation space for each player.
        """

    @abc.abstractmethod
    def action_specs(self) -> PlayerIDToSpec:
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
