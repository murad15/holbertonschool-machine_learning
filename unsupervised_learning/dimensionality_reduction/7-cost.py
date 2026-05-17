#!/usr/bin/env python3
"""t-SNE cost function"""

import numpy as np


def cost(P, Q):
    """
    Calculates the Kullback-Leibler divergence cost for t-SNE.

    Parameters:
    P : numpy.ndarray of shape (n, n)
        P affinities
    Q : numpy.ndarray of shape (n, n)
        Q affinities

    Returns:
    C : float
        Cost of the transformation
    """

    eps = 1e-12

    P_safe = np.clip(P, eps, None)
    Q_safe = np.clip(Q, eps, None)

    C = np.sum(P_safe * np.log(P_safe / Q_safe))

    return C
