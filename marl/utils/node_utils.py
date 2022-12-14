import functools
from ast import Call
from typing import Any, Callable, Optional

import launchpad as lp


def build_courier_node(
    fn: Optional[Callable[..., Any]] = None, disable_run: Optional[bool] = None
) -> Callable[..., lp.CourierNode]:
    """Decorator that deploys a function result onto a `CourierNode'."""

    def _decorate(fn_: Callable[..., Any]):
        @functools.wraps(fn_)
        def _wrapped_fn(*args, **kwargs):
            node = lp.CourierNode(fn_, *args, **kwargs)
            if disable_run:
                node.disable_run()
            return node

        return _wrapped_fn

    # This is required to allow for the decorator be optionally supplied arguments.
    # If the decorator is called with no optional arguments:
    #       @build_courier_node
    #       def caller ...
    # Then the decorator is given `caller` as `fn`, and decorate its.
    if fn:
        return _decorate(fn)
    # Otherwise, if an optional argument is specified:
    #       @build_courier_node(disable_run=True)
    #       def new_caller ...
    # Then the decorator is gven `None` as `fn` instead of `new_caller`. So we return
    # the internal decorator that has closed all optional arguments. Then the internal
    # decorator is given `new_caller`.
    else:
        return _decorate
