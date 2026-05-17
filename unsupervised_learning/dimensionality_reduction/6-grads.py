#!/usr/bin/env python3
"""t-SNE gradients"""

import numpy as np

Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    Calculates gradients of Y for t-SNE.

    Parameters:
    Y : numpy.ndarray of shape (n, ndim)
        Low dimensional representation of X
    P : numpy.ndarray of shape (n, n)
        P affinities

    Returns:
    dY : numpy.ndarray of shape (n, ndim)
        Gradients of Y
    Q : numpy.ndarray of shape (n, n)
        Q affinities
    """

    n, ndim = Y.shape

    # Compute Q affinities and numerator
    Q, num = Q_affinities(Y)

    # Compute pairwise difference tensor (n, n, ndim)
    diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]

    # Compute (P - Q) * num
    PQ = (P - Q) * num

    # Gradient
    dY = np.zeros((n, ndim))
    for i in range(n):
        dY[i] = np.sum(PQ[:, :, np.newaxis] * diff[:, :, :], axis=1)[i]

    return dY, Q
