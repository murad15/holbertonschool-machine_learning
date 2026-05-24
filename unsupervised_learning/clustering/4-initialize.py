#!/usr/bin/env python3
"""Initialize variables for a Gaussian Mixture Model"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initializes variables for a Gaussian Mixture Model."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None

    if type(k) is not int or k <= 0:
        return None, None, None

    n, d = X.shape

    m, _ = kmeans(X, k)

    if m is None:
        return None, None, None

    pi = np.ones(k) / k
    S = np.tile(np.eye(d), (k, 1, 1))

    return pi, m, S
