"""Functions to create and combine hyper parameter sweeps.

References:
 - https://github.com/deepmind/distribution_shift_framework/
"""
import dataclasses
import functools
import itertools
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Type, TypeVar, Union

import six
from ml_collections import config_dict

Sweep = List[config_dict.ConfigDict]


def fixed(parameter_name: str, value: Any) -> Sweep:
  """Creates a sweep for a single parameter/value pair.

  Args:
      parameter_name: The name of the parameter.
      value: The value for the parameter.

  Returns:
      A list containing a single config_dict with the parameter and its value.
  """
  return [_create_config_dict_from_partially_flat({parameter_name: value})]


def sweep(parameter_name: str, values: Iterable[Any]) -> Sweep:
  """Creates a sweep from a list of values for a parameter.

  Args:
      parameter_name: The name of the parameter.
      values: An iterable of values to sweep over for the given parameter.

  Returns:
      A list of config_dicts, each containing the parameter and one of its values.
  """
  return [_create_config_dict_from_partially_flat({parameter_name: value}) for value in values]


def dataclass(config: Any) -> Sweep:
  """Creates a sweep from a dataclass.

  Args:
      config: A dataclass instance.

  Returns:
      A list containing a single config_dict created from the dataclass.

  Raises:
      ValueError: If the input is not a dataclass instance.
  """
  if not dataclasses.is_dataclass(config):
    raise ValueError(f"Expected a dataclass object, got {type(config)}.")
  return [_create_config_dict_from_partially_flat(dataclasses.asdict(config))]


def product(sweeps: Sequence[Sweep]) -> Sweep:
  """Builds a sweep from the cartesian product of a list of sweeps.

  Args:
      sweeps: A sequence of sweeps to combine.

  Returns:
      A sweep built from the cartesian product of the input sweeps.
  """
  if not sweeps:
    return [config_dict.ConfigDict()]
  return [functools.reduce(_combine_parameter_dicts, param_dicts, {}) for param_dicts in itertools.product(*sweeps)]


def zipit(sweeps: Sequence[Sweep]) -> Sweep:
  """Builds a sweep from zipping a list of sweeps.

  Args:
      sweeps: A sequence of sweeps to combine.

  Returns:
      A sweep built from zipping the input sweeps.
  """
  return [functools.reduce(_combine_parameter_dicts, param_dicts, {}) for param_dicts in zip(*sweeps)]


def default(base: Any, overrides: Sweep) -> Sweep:
  """Applies a sweep on top of a set of default values.

  Args:
      base: Base configuration or dataclass specifying default values for each parameter.
      overrides: Sweeps that are treated as overrides to the base/default values.

  Returns:
      A sweep with overrides applied to the base values.
  """

  def _dict_update(x: Dict[str, Any], y: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in x.items():
      if key not in y:
        y[key] = value
    return y

  def _dataclass_update(x: Any, y: Dict[str, Any]) -> Dict[str, Any]:
    return _dict_update(dataclasses.asdict(x), y)

  _update = _dataclass_update if dataclasses.is_dataclass(base) else _dict_update
  return [_update(base, override) for override in overrides]


_T = TypeVar("_T")


def cast(config_type: Type[_T], sweep: Sweep) -> List[_T]:
  """Casts a sweep instance to a type allowing it to be validated.

  The intended use-case for this method is for experiments with a dataclass specifying
  the experiment's settings. A sweep is then validated by attempting to build an instance
  of the settings out of the sweep. Defining a `__post_init__` for said dataclass
  can enable even stronger setting testing.

  NOTE: This method requires type annotations to ensure proper casting.

  Args:
      config_type: The type to cast the sweep instances to.
      sweeps: The sweeps to cast.

  Returns:
      A list of instances of config_type created from the sweeps.
  """
  return [_cast_config(config_type, config) for config in sweep]


def _cast_config(cast_type: Type[_T], config: config_dict.ConfigDict) -> _T:
  """Cast a config as a specific type."""

  # ConfigDict are type strict, so we need a new container to store
  # the modified entries.
  updated_kwargs: Mapping[str, Any] = {}

  for key, value in six.iteritems(config):
    member_type = cast_type.__annotations__[key]

    if isinstance(value, config_dict.ConfigDict):
      updated_kwargs[key] = _cast_config(member_type, value)
      del config[key]

  return cast_type(**config, **updated_kwargs)


def _combine_parameter_dicts(x: Mapping[str, Any], y: Mapping[str, Any]) -> config_dict.ConfigDict:
  """Combine two parameter dictionaries.

  Args:
      x: The first parameter dictionary.
      y: The second parameter dictionary.

  Returns:
      A new config_dict containing the combined parameters from x and y.
  """
  config = _create_config_dict_from_partially_flat(x)
  config.update(_create_config_dict_from_partially_flat(y))
  return config


def _create_config_dict_from_partially_flat(input: Any) -> config_dict.ConfigDict:
  """Create a config-dict from a partially flat dictionary/config-dict.

  Args:
      input: A partially flat dictionary or config_dict.

  Returns:
      A new config_dict created from the input.
  """
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
        # separately and then merge the two configs.
        config[tokens[0]].update(new_children)
      else:
        # Otherwise, the new children define the start of this sub-config.
        config[tokens[0]] = new_children

    else:
      config[key] = _create_config_dict_from_partially_flat(value)

  return config
