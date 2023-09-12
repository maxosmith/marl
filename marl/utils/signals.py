"""Helper methods for handling signals."""

import contextlib
import ctypes
import threading
from typing import Any, Callable, Optional

import launchpad
from absl import logging

_Handler = Callable[[], Any]


@contextlib.contextmanager
def runtime_terminator(callback: Optional[_Handler] = None):
  """Runtime terminator used for stopping computation upon agent termination.

  Runtime terminator optionally executed a provided `callback` and then raises
  `SystemExit` exception in the thread performing the computation.

  Args:
      callback: callback to execute before raising exception.

  Yields:
      None.
  """
  worker_id = threading.get_ident()

  def signal_handler():
    logging.info("Received termination signal.")
    if callback:
      callback()
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(worker_id), ctypes.py_object(SystemExit)
    )
    assert res < 2, "Stopping worker failed"

  launchpad.register_stop_handler(signal_handler)
  yield
  launchpad.unregister_stop_handler(signal_handler)
