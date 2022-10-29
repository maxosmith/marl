"""Functions to create and combine hyper parameter sweeps.

References:
 - https://github.com/deepmind/distribution_shift_framework/
"""
import dataclasses
import functools
import itertools
from typing import Any, Dict, Iterable, List, Sequence, Union

import ml_collections

Config = Union[Dict[str, Any], ml_collections.ConfigDict]
Sweep = List[Config]


def fixed(parameter_name: str, value: Any) -> Sweep:
    """Creates a sweep for a single parameter/value pair."""
    return [{parameter_name: value}]


def sweep(parameter_name: str, values: Iterable[Any]) -> Sweep:
    """Creates a sweep from a list of values for a parameter."""
    return [{parameter_name: value} for value in values]


def dataclass(config: Any) -> Sweep:
    """Creates a sweep from a dataclass."""
    if not dataclasses.is_dataclass(config):
        raise ValueError(f"Expected a dataclass object, got {type(config)}.")
    return [dict(config.__dict__)]


def product(sweeps: Sequence[Sweep]) -> Sweep:
    """Builds a sweep from the cartesian product of a list of sweeps."""
    return [functools.reduce(_combine_parameter_dicts, param_dicts, {}) for param_dicts in itertools.product(*sweeps)]


def zipit(sweeps: Sequence[Sweep]) -> Sweep:
    """Builds a sweep from zipping a list of sweeps."""
    return [functools.reduce(_combine_parameter_dicts, param_dicts, {}) for param_dicts in zip(*sweeps)]


def default(base: Any, overrides: Sequence[Sweep]) -> Sweep:
    """Applies a sweep ontop of a set of default values.

    Args:
        base: Base configuration or dataclass specifying default values for each parameter.
        overrides: Sweeps that are treated as overrides to the base/default values.
    """

    def _dict_update(x, y):
        for key, value in x.items():
            if key not in y:
                y[key] = value
        return y

    def _dataclass_update(x, y):
        return _dict_update(x.__dict__, y)

    _update = _dataclass_update if dataclasses.is_dataclass(base) else _dict_update
    return [_update(base, override) for override in overrides]


def cast(config_type: Any, sweeps: Sequence[Sweep]) -> List[Any]:
    """Casts a sweep instance to a type allowing it to be validated.

    The intended use-case for this method is for experiments with a dataclass specifying
    the experiment's settings. A sweep is then validated by attempting to build an instance
    of the settings out of the sweep. Defining a `__post_init__` for said dataclass
    can enable even stronger setting testing.
    """
    return [config_type(**sweep) for sweep in sweeps]


def _combine_parameter_dicts(x: Dict[str, Any], y: Dict[str, Any]) -> Dict[str, Any]:
    if x.keys() & y.keys():
        raise ValueError(
            "Cannot combine sweeps that set the same parameters. "
            f"Keys in x: {x.keys()}, keys in y: {y.keys}, "
            f"overlap: {x.keys() & y.keys()}"
        )
    return {**x, **y}
