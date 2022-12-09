import time

import launchpad as lp
from absl import logging

from marl.services.counter import Counter
from marl.utils import signals


class StepsLimiter:
    """Process that terminates an experiment when `max_steps` is reached."""

    def __init__(self, counter: Counter, max_steps: int, step_key: str = "actor_steps"):
        """Initializes an instance of a StepsLimiter.

        Args:
            counter: Counter service maintaining global counts across services.
            max_steps: Maximum number of steps to allow before terminating the program.
                NOTE: This is a lower-bound, because we are not garaunteed to check the count
                at every value. Instead, we terminate the program whenver this threshold is crossed.
            step_key: Key associated with the limited step maintained by the counter.
        """
        self._counter = counter
        self._max_steps = max_steps
        self._step_key = step_key

    def run(self):
        """Run steps limiter to terminate an experiment when max_steps is reached."""

        logging.info("StepsLimiter: Starting with max_steps = %d (%s)", self._max_steps, self._step_key)
        with signals.runtime_terminator():
            while True:
                # Update the counts.
                counts = self._counter.get_counts()
                num_steps = counts.get(self._step_key, 0)

                logging.info("StepsLimiter: Reached %d recorded steps", num_steps)

                if num_steps >= self._max_steps:
                    logging.info("StepsLimiter: Max steps of %d was reached, terminating", self._max_steps)
                    lp.stop()

                # Don't spam the counter.
                for _ in range(10):
                    # Do not sleep for a long period of time to avoid LaunchPad program
                    # termination hangs (time.sleep is not interruptible).
                    time.sleep(1)
