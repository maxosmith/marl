"""Common primative types.

This should only contain types built from primitive types, and not
types that require importing structures from within this package.
The package API (`__init__`) should import all of these types and
newly defined types into a single interface.
"""
from typing import Any, Mapping, NamedTuple

import chex

PlayerID = int
# TODO(maxsmith): Move individual into an interface.
Individual = Any

Array = chex.Array
Tree = chex.ArrayTree

# Commonly-used types.
Observation = Tree
Action = Tree
PlayerIDToAction = Mapping[PlayerID, Action]
Params = Tree
NetworkOutput = Tree
QValues = Array
Logits = Array
LogProb = Array
Value = Array
State = Tree


class Transition(NamedTuple):
    """Container for a transition."""

    observation: Tree
    action: Tree
    reward: Tree
    next_observation: Tree
    extras: Tree = ()
