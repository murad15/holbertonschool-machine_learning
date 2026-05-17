#!/usr/bin/env python3
"""Initialize variables for t-SNE"""

import numpy as np


def P_init(X, perplexity):
    """
    Initializes variables required to calculate P affinities in t-SNE.

    Parameters:
    X : numpy.ndarray of shape (n, d)
        Dataset
    perplexity : float
        Desired perplexity

    Returns:
    D : numpy.ndarray of shape (n, n)
        Squared pairwise distance matrix
    P : numpy.ndarray of shape (n, n)
        Initialized P affinities matrix
    betas : numpy.ndarray of shape (n, 1)
        Initialized beta values
    H : float
        Shannon entropy corresponding to perplexity
    """

    n = X.shape[0]

    # Squared norms
    sum_X = np.sum(X ** 2, axis=1)

    # Squared pairwise distance matrix
    D = sum_X[:, np.newaxis] + sum_X - 2 * np.matmul(X, X.T)

    # Remove floating point negatives on diagonal
    np.fill_diagonal(D, 0)

    # Initialize P matrix
    P = np.zeros((n, n))

    # Initialize betas
    betas = np.ones((n, 1))

    # Shannon entropy
    H = np.log2(perplexity)

    return D, P, betas, H
