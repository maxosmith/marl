"""Noop adder."""
from typing import Any, Callable, Optional

from marl import specs, types, worlds


class NoopAdder:
  """An adder interface that does not log data."""

  def add(
      self,
      timestep: worlds.TimeStep,
      action: types.Tree = None,
      extras: types.Tree = (),
  ):
    """Record an action and the following timestep."""
    del timestep, action, extras

  def add_priority_table(
      self, table_name: str, priority_fn: Optional[Callable[..., Any]]
  ):
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
    raise NotImplementedError("Signature not defined for NoopAdder.")
