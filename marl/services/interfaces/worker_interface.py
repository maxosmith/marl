"""Interface for worker nodes that actively execute code.

Workers are placed on CourierNodes and define the `run()` method, which
is run immediately after RPC methods are established. Non-Workers will
idly wait (not work) until their RPC methods are queried.
"""
import abc


class WorkerInterface(abc.ABC):
    """An interface for (potentially) distributed workers."""

    @abc.abstractmethod
    def run(self):
        """Runs the worker."""
