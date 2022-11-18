"""Functions to create and combine hyper parameter sweeps.

References:
 - https://github.com/deepmind/distribution_shift_framework/
"""
import dataclasses
import functools
import itertools
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Union

import six
from ml_collections import config_dict

Sweep = List[config_dict.ConfigDict]


def fixed(parameter_name: str, value: Any) -> Sweep:
    """Creates a sweep for a single parameter/value pair."""
    return [_create_config_dict_from_partially_flat({parameter_name: value})]


def sweep(parameter_name: str, values: Iterable[Any]) -> Sweep:
    """Creates a sweep from a list of values for a parameter."""
    return [_create_config_dict_from_partially_flat({parameter_name: value}) for value in values]


def dataclass(config: Any) -> Sweep:
    """Creates a sweep from a dataclass."""
    if not dataclasses.is_dataclass(config):
        raise ValueError(f"Expected a dataclass object, got {type(config)}.")
    return [_create_config_dict_from_partially_flat(dataclasses.asdict(config))]


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
        return _dict_update(dataclasses.asdict(x), y)

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


def _combine_parameter_dicts(x: Mapping[str, Any], y: Mapping[str, Any]) -> config_dict.ConfigDict:
    """Combine to parameter dictionaries."""
    config = _create_config_dict_from_partially_flat(x)
    config.update(_create_config_dict_from_partially_flat(y))
    return config


def _create_config_dict_from_partially_flat(input: Any) -> config_dict.ConfigDict:
    """Create a config-dict from a partially flat dictionary/config-dict."""
    if (not isinstance(input, config_dict.ConfigDict)) and (not isinstance(input, dict)):
        # Recursive base case.
        return input

    config = config_dict.ConfigDict()

    for key, value in six.iteritems(input):
        if "." in key:
            # This entry is hierarchical.
            tokens = key.split(".")
            new_children = _create_config_dict_from_partially_flat({".".join(tokens[1:]): value})

            if tokens[0] in config:
                # A sub-config already exists, so we need to build-out our new children
                # seperately and then merge the two configs.
                config[tokens[0]].update(new_children)
            else:
                # Otherwise, the new children define the start of this sub-config.
                config[tokens[0]] = new_children

        else:
            config[key] = _create_config_dict_from_partially_flat(value)

    return config
