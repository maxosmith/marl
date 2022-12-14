from typing import Sequence

import numpy as np


def mean_per_second(x: Sequence[float]) -> float:
    return 60.0 / np.mean(x)
