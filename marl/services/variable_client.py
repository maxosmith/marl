"""Variable utilities for JAX."""

import datetime
import time
from concurrent import futures
from typing import List, NamedTuple, Optional, Sequence, Union

import jax
from absl import logging

from marl import _types
from marl.services.interfaces import variable_source_interface


class VariableReference(NamedTuple):
    """Reference to a variable contained in the source."""

    variable_name: str


class ReferenceVariableSource(variable_source_interface.VariableSourceInterface):
    """Variable source which returns references instead of values.

    This is passed to each actor when using a centralized inference server. The
    actor uses this special variable source to get references rather than values.
    These references are then passed to calls to the inference server, which will
    dereference them to obtain the value of the corresponding variables at
    inference time. This avoids passing around copies of variables from each
    actor to the inference server.
    """

    def get_variables(self, names: Sequence[str]) -> List[VariableReference]:
        """Get referneces for all variables."""
        return [VariableReference(name) for name in names]


class VariableClient:
    """A variable client for updating variables from a remote source."""

    def __init__(
        self,
        source: variable_source_interface.VariableSourceInterface,
        key: Optional[Union[str, Sequence[str]]] = None,
        update_period: Union[int, datetime.timedelta] = 1,
        device: Optional[Union[str, jax.xla.Device]] = None,
    ):
        """Initializes the variable client.

        Args:
            client: A variable source from which we fetch variables.
            key: Which variables to request. When multiple keys are used, params
                property will return a list of params. If None, assumes that the
                client has a singular variable set that can be retrieved.
            update_period: Interval between fetches, specified as either (int) a
                number of calls to update() between actual fetches or (timedelta) a time
                interval that has to pass since the last fetch.
            device: The name of a JAX device to put variables on. If None (default),
                VariableClient won't put params on any device.
        """
        logging.info(f"Initializing a VariableClient with source: {source}.")
        self._update_period = update_period
        self._call_counter = 0
        self._last_call = time.time()
        self._source = source
        self._params: Sequence[_types.Params] = None

        self._device = device
        if isinstance(self._device, str):
            self._device = jax.devices(device)[0]

        self._executor = futures.ThreadPoolExecutor(max_workers=1)

        if isinstance(key, str):
            key = [key]

        self._key = key
        self._request = lambda k=key: source.get_variables(k)
        self._future: Optional[futures.Future] = None  # pylint: disable=g-bare-generic
        self._async_request = lambda: self._executor.submit(self._request)

    def update(self, wait: bool = False) -> None:
        """Periodically updates the variables with the latest copy from the source.

        If wait is True, a blocking request is executed. Any active request will be
        cancelled.
        If wait is False, this method makes an asynchronous request for variables.

        Args:
            wait: Whether to execute asynchronous (False) or blocking updates (True).
                Defaults to False.
        """
        # Track calls (we only update periodically).
        self._call_counter += 1

        # Return if it's not time to fetch another update.
        if isinstance(self._update_period, datetime.timedelta):
            if self._update_period.total_seconds() + self._last_call > time.time():
                return
        else:
            if self._call_counter < self._update_period:
                return

        if wait:
            if self._future is not None:
                if self._future.running():
                    self._future.cancel()
                self._future = None
            self._call_counter = 0
            self._last_call = time.time()
            self.update_and_wait()
            return

        # Return early if we are still waiting for a previous request to come back.
        if self._future and not self._future.done():
            return

        # Get a future and add the copy function as a callback.
        self._call_counter = 0
        self._last_call = time.time()
        self._future = self._async_request()
        self._future.add_done_callback(lambda f: self._callback(f.result()))

    def update_and_wait(self):
        """Immediately update and block until we get the result."""
        self._callback(self._request())

    def _callback(self, params_list: List[_types.Params]):
        """Place the parameters on the appropriate device when ready."""
        if self._device and not isinstance(self._source, ReferenceVariableSource):
            # Move variables to a proper device.
            self._params = jax.device_put(params_list, self._device)
        else:
            self._params = params_list

    @property
    def device(self) -> Optional[jax.xla.Device]:
        """Device that variables are placed onto."""
        return self._device

    @property
    def params(self) -> Union[_types.Params, List[_types.Params]]:
        """Returns the first params for one key, otherwise the whole params list."""
        if self._params is None:
            self.update_and_wait()

        if len(self._params) == 1:
            return self._params[0]
        else:
            return self._params
