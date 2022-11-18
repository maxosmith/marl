"""Utility functions for program flags.

NOTE: Flags defined here will not appear when you run "--help" on your program, but instead
only when you run "--helpful". This is because absl distinguishes flags defined in the main
program and all other flags. If you would like flags defined here to show up when you use
"help", you must call `adopt_module_key_flags` on this module. e.g.,
    ```driver.py
    from absl import flags

    from marl.utils import flag_utils


    flags.adopt_module_key_flags(flag_utils)
    ```
"""
from typing import Optional, Sequence

from absl import flags
from ml_collections import config_dict


def common_experiment_flags():
    """Establishes flags that are common across experiments."""
    flags.DEFINE_string(
        name="result_dir",
        default=None,
        help="Directory to write all experimental results.",
        short_name="r",
    )
    flags.DEFINE_multi_string(
        name="overrides",
        default=None,
        help='Configuration override string of the form "key = value".',
        short_name="o",
    )


def parse_overrides(overrides: Sequence[str]) -> config_dict.ConfigDict:
    """Parses override strings into a dictionary of overrides.

    Args:
        overrides: List of override strings of the format: "key = value".

    Returns:
        Dictionary representation of x {key: value}, with an attempt to
        convert the type
    """
    config = config_dict.ConfigDict()

    for override in overrides:
        tokens = override.split(" = ")
        config[tokens[0]] = " = ".join(tokens[1:])

    return config


def apply_overrides(overrides: Sequence[str], base: Optional[config_dict.ConfigDict]) -> config_dict.ConfigDict:
    """Applies override strings to a configuration.

    Args:
        overrides: List of override strings of the format: "key = value".
        base: Base configuration file.

    Returns:
        Base configuration with overrides applied.
    """
    overrides = parse_overrides(overrides=overrides)

    for key, value in overrides.items():
        # Attempt to cast the string-representation of the value to the correct type.
        if base[key] is not None:
            value = type(base[key])(value)
        base[key] = value

    return base
