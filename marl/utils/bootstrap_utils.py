""" Utility functions for performing statistical bootstrapping operations.

References:
 - https://arch.readthedocs.io/en/latest/bootstrap/confidence-intervals.html
 - https://github.com/google-research/rliable/blob/master/library.py
"""
from typing import Callable, Sequence, Tuple

import arch.bootstrap as arch_bs
import numpy as np


def confidence_intervals(
    x: np.ndarray, fn: Callable, method: str = "percentile", size: float = 0.95, reps: int = 1000
) -> Sequence[float]:
    """Computes interval estimates via bootstrapping.

    Args:
        x: Scores.
        fn: Function that computes the aggregate performance.
        method: Bootstrapping method one of `basic', `percentile', `studentized', `norm' (identical to `var', `cov'),
            `bc' (identical to `debiased', `bias-corrected'), or `bca'
        size: Coverage of confidence interval.
        reps: Number of bootstrap replications.

    Returns:
        Computed confidence interval.
    """
    x = np.array(x)
    assert len(x.shape) == 1, "Only defined for single dimension variables."
    assert (0 < size) and (size < 1), "Interval size must be between (0, 1)."
    assert reps > 0, "Number of replications must be great than zero."

    bs = arch_bs.IIDBootstrap(x)
    ci = bs.conf_int(func=fn, reps=reps, method=method, size=size)
    return ci


def get_point_and_interval_estimates(
    x: np.ndarray,
    fn: Callable,
    method: str = "percentile",
    size: float = 0.95,
    reps: int = 1000,
) -> Tuple[float, np.ndarray]:
    """Computes point and intevrval estimates via bootstrapping along the last axis of `x`.

    Args:
        x: Scores.
        fn: Function that computes the aggregate performance.
        method: Bootstrapping method one of `basic', `percentile', `studentized', `norm' (identical to `var', `cov'),
            `bc' (identical to `debiased', `bias-corrected'), or `bca'
        size: Coverage of confidence interval.
        reps: Number of bootstrap replications.

    Returns:
        Tuple containing the point estimate then interval estimate.
    """
    if len(x.shape) == 1:
        point_estimate = fn(x)
        interval_estimate = confidence_intervals(x=x, fn=fn, method=method, size=size, reps=reps)
        return point_estimate, interval_estimate
    else:
        # Apply point and interval estimates across last axis.
        point_estimates = np.zeros(x.shape[:-1])
        interval_estimates = np.zeros(list(x.shape[:-1]) + [2])

        for slice in np.ndindex(point_estimates.shape):
            point_estimates[slice] = fn(x[slice])
            interval_estimates[slice] = np.ravel(
                confidence_intervals(x=x[slice], fn=fn, method=method, size=size, reps=reps)
            )

        return point_estimates, interval_estimates
