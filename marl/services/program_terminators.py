import time

import launchpad as lp
from absl import logging

from marl.services.counter import Counter
from marl.utils import signals


class StepsLimiter:
    """Process that terminates an experiment when `max_steps` is reached."""

    def __init__(self, counter: Counter, max_steps: int, steps_key: str = "actor_steps"):
        self._counter = counter
        self._max_steps = max_steps
        self._steps_key = steps_key

    def run(self):
        """Run steps limiter to terminate an experiment when max_steps is reached."""

        logging.info("StepsLimiter: Starting with max_steps = %d (%s)", self._max_steps, self._steps_key)
        with signals.runtime_terminator():
            while True:
                # Update the counts.
                counts = self._counter.get_counts()
                num_steps = counts.get(self._steps_key, 0)

                logging.info("StepsLimiter: Reached %d recorded steps", num_steps)

                if num_steps > self._max_steps:
                    logging.info("StepsLimiter: Max steps of %d was reached, terminating", self._max_steps)
                    lp.stop()

                # Don't spam the counter.
                for _ in range(10):
                    # Do not sleep for a long period of time to avoid LaunchPad program
                    # termination hangs (time.sleep is not interruptible).
                    time.sleep(1)
