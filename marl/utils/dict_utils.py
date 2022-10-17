from typing import Any, Mapping


def prefix_keys(x: Mapping[str, Any], prefix: str) -> Mapping[str, Any]:
    """Apply a prefix to all dictionary keys."""
    return {f"{prefix}{k}": v for k, v in x.items()}
