"""Service for stopping a running `LaunchPad` program."""
import time

import launchpad as lp
from absl import logging

from marl.services import interfaces
from marl.utils import signals


class ProgramTerminator(interfaces.Worker):
    """Process that provides an interface for stopping a program.

    This service is useful in situations where the overall runtime
    of a program is not known in advance. This service can then allow
    the program launcher, or another service, to dynamically stop
    the entire program (instead of just themselves).
    """

    def __init__(self):
        """Initializer."""
        self._should_terminate = False

    def stop(self):
        """Stops the program."""
        logging.info("Stop requested.")
        self._should_terminate = True

    def run(self):
        """Run the service, waiting for call to `stop()`."""
        with signals.runtime_terminator():
            while True:
                time.sleep(1)
                if self._should_terminate:
                    logging.info("Terminating.")
                    lp.stop()
