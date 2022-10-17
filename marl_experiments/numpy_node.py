from typing import Optional

import launchpad as lp
import numpy as np
from absl import app


class Logger:
    def __init__(self, data: Optional[np.ndarray] = None) -> None:
        self._data = data

    def receive(self, x: np.ndarray) -> None:
        print(x)

    def send(self):
        return self._data


def main(_):
    program = lp.Program("test")
    logger_handle = program.add_node(lp.CourierNode(Logger, np.zeros([3])), label="logger")
    lp.launch(program, launch_type=lp.LaunchType.LOCAL_MULTI_THREADING)

    logger_client = logger_handle.dereference()

    # logger_client.receive(np.ones_like([3]))  # Hangs indefinitely
    print(logger_client.send())  # --> [0, 0, 0]


if __name__ == "__main__":
    app.run(main)
