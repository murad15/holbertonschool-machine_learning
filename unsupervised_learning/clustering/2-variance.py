#!/usr/bin/env python3
"""Calculate intra-cluster variance"""

import numpy as np


def variance(X, C):
    """Calculates the total intra-cluster variance for a data set."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None

    if X.shape[1] != C.shape[1]:
        return None

    distances = np.sum((X[:, np.newaxis] - C) ** 2, axis=2)

    return np.sum(np.min(distances, axis=1))
