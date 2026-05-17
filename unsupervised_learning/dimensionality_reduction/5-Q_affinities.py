#!/usr/bin/env python3
"""Calculates Q affinities"""

import numpy as np


def Q_affinities(Y):
    """
    Calculates the Q affinities.

    Parameters:
    Y : numpy.ndarray of shape (n, ndim)
        Low dimensional representation of X

    Returns:
    Q : numpy.ndarray of shape (n, n)
        Q affinities
    num : numpy.ndarray of shape (n, n)
        Numerator of the Q affinities
    """

    # Squared norms
    sum_Y = np.sum(Y ** 2, axis=1)

    # Squared pairwise distances
    D = sum_Y[:, np.newaxis] + sum_Y - 2 * np.matmul(Y, Y.T)

    # Numerator matrix
    num = 1 / (1 + D)

    # Zero diagonal
    np.fill_diagonal(num, 0)

    # Q affinities
    Q = num / np.sum(num)

    return Q, num
