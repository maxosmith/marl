"""Utilities for testing arenas."""
import dataclasses
from typing import Any, Callable, Optional, Sequence, Tuple

from marl import individuals, specs, types, worlds
from marl.utils import loggers, tree_utils

TWO_STEP_TRAJ = (
    {
        0: worlds.TimeStep(worlds.StepType.FIRST, 0.5, 1),
        1: worlds.TimeStep(worlds.StepType.FIRST, 0, 1),
    },
    {0: 4, 1: 5},
    {
        0: worlds.TimeStep(worlds.StepType.MID, 0.2, 2),
        1: worlds.TimeStep(worlds.StepType.MID, 0.25, 2),
    },
    {0: 7, 1: 8},
    {
        0: worlds.TimeStep(worlds.StepType.LAST, 1, 3),
        1: worlds.TimeStep(worlds.StepType.LAST, 0, 3),
    },
)


class MockGame(worlds.Game):
  """Mock game for testing.

  See `worlds.Game` for extended API descriptions.

  Args:
    trajectory: Sequence of timesteps that the game will produce, starting with the
      timestep produced at reset.
  """

  def __init__(
      self,
      trajectory: Sequence[worlds.PlayerIDToTimestep],
      expected_actions: Sequence[types.Action],
      *,
      reward_specs: Optional[None] = None,
      observation_specs: Optional[None] = None,
      action_specs: Optional[None] = None,
  ):
    """Initializer."""
    super().__init__()
    self._trajectory = trajectory
    self._expected_actions = expected_actions
    self._reward_specs = reward_specs
    self._observation_specs = observation_specs
    self._action_specs = action_specs
    self._t = None

  def reset(self) -> worlds.PlayerIDToTimestep:
    """Starts a new sequence and returns the first `TimeStep` of this sequence."""
    self._t = 1
    return self._trajectory[0]

  def step(self, actions: types.PlayerIDToAction) -> worlds.PlayerIDToTimestep:
    """Updates the environment according to the action and returns a `TimeStep`."""
    if self._t is None:
      raise RuntimeError("`reset` must be called before `step`.")
    if self._t > len(self._trajectory):
      raise RuntimeError(
          f"Called step on mock game {self._time} times, when only {len(self._trajectory)} " "timesteps were provided."
      )
    # Compare with t-1 actions, because reset increments time.
    tree_utils.assert_equals(actions, self._expected_actions[self._t - 1])
    timesteps = self._trajectory[self._t]
    self._t += 1
    return timesteps

  def reward_specs(self) -> specs.PlayerIDToSpec:
    """Describes the reward returned by the game to each player."""
    if self._reward_specs is None:
      raise RuntimeError("Reward Specs were not defined for this stub.")
    return self._reward_specs

  def observation_specs(self) -> specs.PlayerIDToSpec:
    """Defines the observations provided by the game to each player."""
    if self._observation_specs is None:
      raise RuntimeError("Observation Specs were not defined for this stub.")
    return self._observation_specs

  def action_specs(self) -> specs.PlayerIDToSpec:
    """Defines the actions that should be provided to `step` by each player."""
    if self._action_specs is None:
      raise RuntimeError("Action Specs were not defined for this stub.")
    return self._action_specs


@dataclasses.dataclass
class MockAdder:
  """Mock adder that verifies data added to replay for tests."""

  expected_adds: Sequence[Tuple[worlds.TimeStep, types.Tree, types.Tree]]

  def __post_init__(self):
    """Post initializer."""
    self._t = 0

  def add(
      self,
      timestep: worlds.TimeStep,
      action: types.Tree = None,
      extras: types.Tree = (),
  ):
    """Record an action and the following timestep."""
    expected_ts, expected_action, expected_extras = self.expected_adds[self._t]
    self._t += 1
    tree_utils.assert_equals(timestep, expected_ts)
    tree_utils.assert_equals(action, expected_action)
    tree_utils.assert_equals(extras, expected_extras)

  def add_priority_table(self, table_name: str, priority_fn: Optional[Callable[..., Any]]):
    """Add a priority function for sampling from a table."""
    del table_name, priority_fn

  def reset(self):
    """Resets the adder's buffer."""
    pass

  @classmethod
  def signature(
      cls,
      environment_spec: specs.EnvironmentSpec,
      extras_spec=(),
      sequence_length: Optional[int] = None,
  ):
    """This is a helper method for generating signatures for Reverb tables."""
    del cls, environment_spec, extras_spec, sequence_length
    raise NotImplementedError("Signature not defined for MockAdder.")


@dataclasses.dataclass
class MockLogger:
  """Mock logger that verifies logged data for tests."""

  expected_logs: Sequence[loggers.LogData]

  def __post_init__(self):
    """Post initializer."""
    self._t = 0

  def write(self, data: loggers.LogData):
    """Writes `data` to destination (file, terminal, database, etc.)."""
    tree_utils.assert_equals(data, self.expected_logs[self._t])
    self._t += 1

  def close(self):
    """Closes the logger and underlying services."""
    pass


class MockActionSequenceBot(individuals.Bot):
  """`ActionSequenceBot` that tracks update calls."""

  def __init__(
      self,
      sequence: Sequence[types.Action],
      expected_timesteps: Sequence[worlds.TimeStep],
  ) -> None:
    super().__init__()
    self._sequence = sequence
    self._sequence_len = len(self._sequence)
    self._expected_timesteps = expected_timesteps
    self._t = None

  def step(
      self,
      state: types.State,
      timestep: worlds.TimeStep,
  ) -> Tuple[types.Action, types.State]:
    """Selects an action to take given the current timestep."""
    if self._t is None:
      raise RuntimeError("`episode_reset` must be called before `step`.")
    tree_utils.assert_equals(timestep, self._expected_timesteps[self._t])
    action = self._sequence[self._t % self._sequence_len]
    self._t += 1
    return state, action

  def episode_reset(self, timestep: worlds.TimeStep):
    """Resets the agent's episodic state."""
    del timestep
    self._t = 0
    return ()


class SpyActionSequenceAgent(individuals.Agent):
  """`ActionSequenceBot` that tracks update calls."""

  def __init__(self, sequence: Sequence[types.Action]) -> None:
    super().__init__()
    self._sequence = sequence
    self._sequence_len = len(self._sequence)
    self._t = None
    self.update_calls = 0

  def step(
      self,
      state: types.State,
      timestep: worlds.TimeStep,
  ) -> Tuple[types.Action, types.State]:
    """Selects an action to take given the current timestep."""
    del timestep
    if self._t is None:
      raise RuntimeError("`episode_reset` must be called before `step`.")
    action = self._sequence[self._t % self._sequence_len]
    self._t += 1
    return state, action

  def update(self) -> None:
    """Maybe update the agent's internal parameters."""
    self.update_calls += 1

  def episode_reset(self, timestep: worlds.TimeStep):
    """Resets the agent's episodic state."""
    del timestep
    self._t = 0
    return ()
