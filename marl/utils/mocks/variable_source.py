import threading
from typing import List, Optional, Sequence

from marl import _types, services


class VariableSource(services.VariableSourceInterface):
    """Fake variable source."""

    def __init__(
        self,
        variables: Optional[_types.Tree] = None,
        barrier: Optional[threading.Barrier] = None,
        use_default_key: bool = True,
    ):
        # Add dummy variables so we can expose them in get_variables.
        if use_default_key:
            self._variables = {"policy": [] if variables is None else variables}
        else:
            self._variables = variables
        self._barrier = barrier

    def get_variables(self, names: Sequence[str]) -> List[_types.Tree]:
        if self._barrier is not None:
            self._barrier.wait()
        return [self._variables[name] for name in names]
