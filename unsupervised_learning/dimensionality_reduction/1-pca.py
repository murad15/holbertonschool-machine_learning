#!/usr/bin/env python3
"""PCA transformation"""

import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset.

    Parameters:
    X : numpy.ndarray of shape (n, d)
        Dataset
    ndim : int
        New dimensionality

    Returns:
    T : numpy.ndarray of shape (n, ndim)
        Transformed version of X
    """

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(X)

    # Projection matrix
    W = Vt.T[:, :ndim]

    # Transform data
    T = np.matmul(X, W)

    return T
