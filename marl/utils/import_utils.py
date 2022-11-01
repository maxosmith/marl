import importlib
from typing import Any


def str_to_class(path: str) -> Any:
    tokens = path.split(".")
    module = ".".join(tokens[:-1])
    name = tokens[-1]
    module = importlib.import_module(module)
    return getattr(module, name)


def initialize(ctor_path, kwargs):
    return str_to_class(ctor_path)(**kwargs)
