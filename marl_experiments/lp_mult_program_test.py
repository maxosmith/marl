"""This example presents clean node termination."""


import threading
import time

import launchpad as lp
from absl import app, logging


def _sleep():
    while not lp.wait_for_stop(1):
        logging.info("Sleeping again...")
    logging.info("Clean termination of _sleep node")
    time.sleep(2)


def _wait_for_stop():
    lp.wait_for_stop()
    logging.info("Clean termination of _wait_for_stop node")
    time.sleep(2)


def _stop_event():
    lp.stop_event().wait()
    logging.info("Clean termination of _stop_event node")
    time.sleep(2)


def _register_stop_handler():
    """Showcases the use of lp.register_stop_handler."""
    stop = threading.Event()

    def _stop_handler():
        logging.info("_stop_handler called")
        stop.set()

    lp.register_stop_handler(_stop_handler)
    stop.wait()
    logging.info("Clean termination of _register_stop_handler node")
    time.sleep(2)


def _stop_program():
    time.sleep(4)
    lp.stop()


def make_program() -> lp.Program:
    """Define the distributed program topology."""
    program = lp.Program("program_wait")
    program.add_node(lp.CourierNode(_sleep), label="sleep")
    program.add_node(lp.CourierNode(_wait_for_stop), label="_wait_for_stop")
    program.add_node(lp.CourierNode(_stop_event), label="_stop_event")
    program.add_node(lp.CourierNode(_register_stop_handler), label="_register_stop_handler")
    program.add_node(lp.CourierNode(_stop_program), label="_stop_program")
    return program


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    program = make_program()
    controller = lp.launch(program)
    if not controller:
        logging.info("Waiting for program termination is not supported.")
        return
    controller.wait()
    logging.info("Program finished.")

    program = make_program()
    controller = lp.launch(program)
    if not controller:
        logging.info("Waiting for program termination is not supported.")
        return
    controller.wait()
    logging.info("Program finished.")


if __name__ == "__main__":
    app.run(main)
