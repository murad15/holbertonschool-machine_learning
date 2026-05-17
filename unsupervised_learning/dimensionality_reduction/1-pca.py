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

    # Center the data
    X = X - np.mean(X, axis=0)

    # SVD
    _, _, Vt = np.linalg.svd(X)

    # Projection matrix
    W = Vt.T[:, :ndim]

    # Transform the data
    T = np.matmul(X, W)

    return T
