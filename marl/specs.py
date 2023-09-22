"""Specify the interface to operations."""
from typing import Any, Iterable, Mapping, NamedTuple, Union

import tensorflow as tf
from dm_env import specs as dm_env_specs

from marl import types

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


GameSpec = Mapping[types.PlayerID, EnvironmentSpec]

# Game specs.
PlayerIDToSpec = Mapping[types.PlayerID, TreeSpec]
PlayerIDToEnvSpec = Mapping[types.PlayerID, EnvironmentSpec]
