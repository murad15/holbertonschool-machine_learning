#!/usr/bin/env python3
"""K-means clustering"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """Performs K-means clustering on a dataset."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    if type(k) is not int or k <= 0:
        return None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None

    n, d = X.shape
    minimum = X.min(axis=0)
    maximum = X.max(axis=0)

    C = np.random.uniform(minimum, maximum, size=(k, d))

    for _ in range(iterations):
        C_prev = C.copy()

        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        for i in range(k):
            points = X[clss == i]

            if points.shape[0] == 0:
                C[i] = np.random.uniform(minimum, maximum, size=d)
            else:
                C[i] = points.mean(axis=0)

        if np.array_equal(C, C_prev):
            return C, clss

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(distances, axis=1)

    return C, clss
