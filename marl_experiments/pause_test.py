import time

import launchpad as lp
from absl import app, logging


class Worker:
    def run(self):
        self._pause = False

        while True:
            # Pretend to do work.
            logging.info("Work!")
            time.sleep(0.1)

            if self._pause:
                break

    def stop(self):
        self._pause = True


def main(_):

    program = lp.Program(name="test")

    worker_handle = program.add_node(lp.CourierNode(Worker), label="worker")

    lp.launch(program, launch_type=lp.LaunchType.LOCAL_MULTI_PROCESSING, terminal="current_terminal")

    worker_handle.dereference().stop()


if __name__ == "__main__":
    app.run(main)
