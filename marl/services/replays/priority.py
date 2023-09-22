from typing import Callable, Mapping, NamedTuple, Optional

from marl import types

DEFAULT_PRIORITY_TABLE = "priority_table"


class PriorityFnInput(NamedTuple):
  """The input to a priority function consisting of stacked steps."""

  observations: types.Tree
  actions: types.Tree
  rewards: types.Tree
  start_of_episode: types.Tree
  extras: types.Tree


# Define the type of a priority function and the mapping from table to function.
PriorityFn = Callable[["PriorityFnInput"], float]
PriorityFnMapping = Mapping[str, Optional[PriorityFn]]
