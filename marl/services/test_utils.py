"""Test utilities."""
import dataclasses
from typing import List, Optional, Sequence, Union

import jax

from marl import types
from marl.services import interfaces, variable_client


@dataclasses.dataclass
class StubVariableClient(variable_client.VariableClient):
  """Variable client that provides a predefined set of expected params."""

  returned_variables: Sequence[List[types.Params]]

  def __post_init__(self):
    """Post initializer."""
    self._t = 0

  def update(self, wait: bool = False):
    """Periodically updates the variables with the latest copy from the source."""
    del wait
    self._t += 1

  def update_and_wait(self):
    """Immediately update and block until we get the result."""
    raise NotImplementedError()

  @property
  def device(self) -> Optional[jax.Device]:
    """Device that variables are placed onto."""
    raise NotImplementedError()

  @property
  def params(self) -> Union[types.Params, List[types.Params]]:
    """Returns the first params for one key, otherwise the whole params list."""
    return (self.returned_variables[self._t], self._t)


@dataclasses.dataclass
class StubVariableSource(interfaces.VariableSource):
  """Abstract source of variables."""

  returned_variables: Sequence[List[types.Tree]]

  def __post_init__(self):
    """Post initializer."""
    self._t = 0

  def get_variables(self, names: Sequence[str] | None = None) -> List[types.Tree]:
    """Return the named variables as a collection of (nested) numpy arrays."""
    if self._t > len(self.returned_variables):
      raise RuntimeError(
          f"Called `get_variables` {self._t} times when only {len(self.returned_variables)} calls are expected."
      )
    self._t += 1
    return self.returned_variables[self._t - 1]
