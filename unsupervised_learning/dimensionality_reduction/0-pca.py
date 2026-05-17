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
        Weights matrix that maintains var fraction of X's original var
    """

    # SVD of X
    _, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Cumulative explained variance ratio
    cumsum = np.cumsum(S ** 2)

    # Minimum dimensions needed
    nd = np.searchsorted(cumsum / cumsum[-1], var) + 1

    # Weights matrix
    W = Vt[:nd].T

    return W
