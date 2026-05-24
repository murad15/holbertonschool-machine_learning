#!/usr/bin/env python3
"""qwqdm qwd qwdas asf asfsaf"""


import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    if type(k) is not int or k <= 0:
        return None

    return np.random.uniform(
        X.min(axis=0),
        X.max(axis=0),
        size=(k, X.shape[1])
    )
