from typing import Any, List, Tuple

import numpy as np


def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    """Smooth scalars by their exponential moving average.

    Args:
        scalars: List of data.
        weight: Smoothing weight.

    Returns:
        Smoothed data.
    """
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed


def trim_ragged_runs(xs: List[List[Any]], ys: List[List[Any]]) -> Tuple[List[Any], List[List[Any]]]:
    """Trim runs of unequal length (ragged) to the size of the shortest run.

    Args:
        xs: X values for each run. It is assumed that all of the runs have their x-values aligned.
        ys: Y values for each run.

    Returns:
        Tuple containing:
            * The new xs, which are the xs corresponding to the shortest run.
            * The new ys, which are a non-ragged tensor containing each run's results trimmed
              to the length of the shortest run.
    """
    shortest_run = np.argmin([len(x) for x in xs])
    new_xs = np.array(xs[shortest_run])
    new_ys = np.array([y[: len(new_xs)] for y in ys])
    return new_xs, new_ys
