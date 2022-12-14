import time

import launchpad as lp
from absl import logging

from marl.utils import signals


class Stopper:
    """Process that provides an interface for stopping a program."""

    def __init__(self):
        self._should_terminate = False

    def stop(self):
        """Stops the program."""
        logging.info("Stopper: stop requested.")
        self._should_terminate = True

    def run(self):
        with signals.runtime_terminator():
            while True:
                time.sleep(1)
                if self._should_terminate:
                    logging.info("Stopper: terminating.")
                    lp.stop()
