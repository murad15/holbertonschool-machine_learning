#!/usr/bin/env python3
"""PCA function"""

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.

    Parameters:
    X : numpy.ndarray of shape (n, d)
        Dataset where each column has a mean of 0

    var : float
        Fraction of variance that the PCA transformation should maintain

    Returns:
    W : numpy.ndarray of shape (d, nd)
        Weights matrix that maintains var fraction of X's original variance
    """

    # Singular Value Decomposition
    _, S, Vt = np.linalg.svd(X)

    # Compute cumulative variance ratio
    cumsum = np.cumsum(S)

    # Find minimum number of dimensions
    nd = np.searchsorted(cumsum / cumsum[-1], var) + 1

    # Principal components
    W = Vt.T[:, :nd]

    return W
