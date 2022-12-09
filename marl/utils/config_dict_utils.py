from typing import List

from ml_collections import config_dict


def flatten_keys(input: config_dict.ConfigDict, delimiter: str = ".") -> List[str]:
    def _flatten_keys(node: config_dict.ConfigDict, prefix: str):
        flat_keys = []

        for key, value in node.items():
            key = f"{prefix}{delimiter}{key}" if prefix else key

            if isinstance(value, config_dict.ConfigDict):
                flat_keys.extend(_flatten_keys(value, prefix=key))
            else:
                flat_keys.append(key)

        return flat_keys

    return _flatten_keys(input, prefix="")


def key_in(key: str, input: config_dict.ConfigDict, delimiter: str = ".") -> bool:
    """Checks if a key is in a config recursively."""
    tokens = key.split(delimiter)

    for token in tokens:
        if token in input:
            input = input[token]
        else:
            return False

    return True
