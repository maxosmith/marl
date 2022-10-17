"""Specs: objects that describe the interaction API of a game/env."""
import operator
from typing import Any, Iterable, Mapping, NamedTuple, Union

import numpy as np
import tensorflow as tf
import tree

from marl import _types, worlds


def make_environment_spec(environment: worlds.Environment) -> worlds.EnvironmentSpec:
    """Returns an `EnvironmentSpec` describing values used by an environment."""
    return worlds.EnvironmentSpec(
        observation=environment.observation_spec(),
        action=environment.action_spec(),
        reward=environment.reward_spec(),
    )


def make_game_specs(game: worlds.Game) -> worlds.PlayerIDToEnvSpec:
    """Returns an `EnvironmentSpec` describing values used by an environment."""
    obs_specs = game.observation_specs()
    action_specs = game.action_specs()
    reward_specs = game.reward_specs()

    agent_to_spec = {}
    for player_id in obs_specs.keys():
        agent_to_spec[player_id] = worlds.EnvironmentSpec(
            observation=obs_specs[player_id],
            action=action_specs[player_id],
            reward=reward_specs[player_id],
        )
    return agent_to_spec


def assert_equal_tree_specs(x: worlds.TreeSpec, y: worlds.TreeSpec) -> None:
    """Asser5ts that two trees contain the same specs and structure.

    Args:
        x: a tree of specs.
        y: a tree of specs.

    Raises:
        ValueError: if the two structures differ.
        TypeError: if the two structures differ in their type.
    """
    tree.assert_same_structure(x, y)
    assert np.all(tree.flatten(tree.map_structure(operator.eq, x, y)))


def make_tree_spec(data: _types.Tree) -> worlds.TreeSpec:
    """Makes a spec describing a tree."""

    def _build_spec(x: _types.Array) -> worlds.ArraySpec:
        return worlds.ArraySpec(x.shape, x.dtype)

    return tree.map_structure(_build_spec, data)


def validate_spec(spec: worlds.TreeSpec, value: _types.Tree):
    """Validate a value from a potentially nested spec."""
    tree.assert_same_structure(value, spec)
    tree.map_structure(lambda s, v: s.validate(v), spec, value)


def generate_from_spec(spec: worlds.TreeSpec) -> _types.Tree:
    """Generate a value from a potentially nested spec."""

    def _normalize_array(array: worlds.ArraySpec) -> worlds.ArraySpec:
        """Converts bounded arrays with (-inf,+inf) bounds to unbounded arrays.

        The returned array should be mostly equivalent to the input, except that
        `generate_value()` returns -infs on arrays bounded to (-inf,+inf) and zeros
        on unbounded arrays.

        Args:
            array: the array to be normalized.

        Returns:
            normalized array.
        """
        if isinstance(array, worlds.DiscreteArraySpec):
            return array
        if not isinstance(array, worlds.BoundedArraySpec):
            return array
        if not (array.minimum == float("-inf")).all():
            return array
        if not (array.maximum == float("+inf")).all():
            return array
        return worlds.ArraySpec(array.shape, array.dtype, array.name)

    return tree.map_structure(lambda s: _normalize_array(s).generate_value(), spec)


def generate_from_tf_spec(spec: worlds.TreeTFSpec) -> _types.Tree:
    """Generate a value defined from a structure of `TensorSpec`s."""

    def _to_spec(spec: tf.TensorSpec) -> worlds.ArraySpec:
        return worlds.ArraySpec(spec.shape, spec.dtype.as_numpy_dtype(), spec.name)

    return tree.map_structure(lambda x: _to_spec(x).generate_value(), spec)


def zeros_from_spec(spec: worlds.TreeSpec) -> _types.Tree:
    """Generate an instance of a spec containing all zeros."""
    return tree.map_structure(lambda x: np.zeros(x.shape, x.dtype), spec)


def get_player_spec(env_spec: worlds.EnvironmentSpec, player_id: _types.PlayerID) -> worlds.EnvironmentSpec:
    """Get a single player's spec from the environment spec."""
    return worlds.EnvironmentSpec(
        action=env_spec.action[player_id],
        discount=env_spec.discount,  # Discounts are assumed to be shared amongst agents.
        observation=env_spec.observation[player_id],
        reward=env_spec.reward[player_id],
    )
