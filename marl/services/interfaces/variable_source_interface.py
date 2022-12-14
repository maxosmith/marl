"""Interface for sources of variables."""
import abc
from typing import List, Optional, Sequence

from marl import _types


class VariableSourceInterface(abc.ABC):
    """Abstract source of variables.

    Objects which implement this interface provide a source of variables, returned
    as a collection of (nested) numpy arrays. Generally this will be used to
    provide variables to some learned policy/etc.
    """

    @abc.abstractmethod
    def get_variables(self, names: Optional[Sequence[str]] = None) -> List[_types.Tree]:
        """Return the named variables as a collection of (nested) numpy arrays.

        Args:
            names: args where each name is a string identifying a predefined subset of
                the variables.

        Returns:
            A list of (nested) numpy arrays `variables` such that `variables[i]`
            corresponds to the collection named by `names[i]`.
        """
