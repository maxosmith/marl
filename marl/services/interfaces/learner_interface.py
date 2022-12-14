"""Service defining a learning loop."""
import abc
import itertools
from typing import List, Optional, Sequence

from marl import _types
from marl.services.interfaces import (saveable_interface,
                                      variable_source_interface,
                                      worker_interface)
from marl.utils import distributed_utils


class LearnerInterface(
    variable_source_interface.VariableSourceInterface,
    worker_interface.WorkerInterface,
    saveable_interface.SaveableInterface,
):
    """Abstract learner object.

    This corresponds to an object which implements a learning loop. A single step
    of learning should be implemented via the `step` method and this step
    is generally interacted with via the `run` method which runs update
    continuously.

    All objects implementing this interface should also be able to take in an
    external dataset (see acme.datasets) and run updates using data from this
    dataset. This can be accomplished by explicitly running `learner.step()`
    inside a for/while loop or by using the `learner.run()` convenience function.
    Data will be read from this dataset asynchronously and this is primarily
    useful when the dataset is filled by an external process.
    """

    @abc.abstractmethod
    def update(self):
        """Perform an update step of the learner's parameters."""

    def run(self, num_steps: Optional[int] = None) -> None:
        """Run the update loop; typically an infinite loop which calls step."""

        iterator = range(num_steps) if num_steps is not None else itertools.count()

        for _ in iterator:
            self.update()

    def save(self):
        raise NotImplementedError('Method "save" is not implemented.')

    def restore(self, state):
        raise NotImplementedError('Method "restore" is not implemented.')

    def get_variables(self, names: Sequence[str]) -> List[_types.Tree]:
        """Return the named variables as a collection of (nested) numpy arrays.

        Args:
            names: args where each name is a string identifying a predefined subset of
                the variables.

        Returns:
            A list of (nested) numpy arrays `variables` such that `variables[i]`
            corresponds to the collection named by `names[i]`.
        """
        return [distributed_utils.get_from_first_device(self._state.params, as_numpy=False)]
