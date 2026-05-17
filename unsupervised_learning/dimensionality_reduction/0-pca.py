#!/usr/bin/env python3
"""PCA function"""

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.

    Parameters:
    X : numpy.ndarray of shape (n, d)
        Dataset where:
        - n is the number of data points
        - d is the number of dimensions
        - all dimensions have a mean of 0

    var : float
        Fraction of variance to preserve

    Returns:
    W : numpy.ndarray of shape (d, nd)
        Weights matrix that preserves the required variance
    """

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Compute explained variance ratio
    explained_variance = (S ** 2)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)

    # Cumulative variance
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Find minimum number of dimensions to preserve `var`
    nd = np.searchsorted(cumulative_variance, var) + 1

    # Principal components
    W = Vt[:nd].T

    return W
