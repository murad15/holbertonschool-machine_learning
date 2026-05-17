#!/usr/bin/env python3
"""Calculates symmetric P affinities"""

import numpy as np

P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a dataset.

    Parameters:
    X : numpy.ndarray of shape (n, d)
        Dataset
    tol : float
        Maximum tolerance for Shannon entropy difference
    perplexity : float
        Desired perplexity

    Returns:
    P : numpy.ndarray of shape (n, n)
        Symmetric P affinities matrix
    """

    n, _ = X.shape

    # Initialize variables
    D, P, betas, H = P_init(X, perplexity)

    for i in range(n):
        # Exclude self-distance
        Di = np.concatenate((D[i, :i], D[i, i + 1:]))

        beta = betas[i]
        beta_min = None
        beta_max = None

        # Compute initial entropy and affinities
        Hi, Pi = HP(Di, beta)

        # Binary search for correct beta
        Hdiff = Hi - H

        while np.abs(Hdiff) > tol:
            if Hdiff > 0:
                beta_min = beta.copy()

                if beta_max is None:
                    beta *= 2.
                else:
                    beta = (beta + beta_max) / 2.
            else:
                beta_max = beta.copy()

                if beta_min is None:
                    beta /= 2.
                else:
                    beta = (beta + beta_min) / 2.

            Hi, Pi = HP(Di, beta)
            Hdiff = Hi - H

        # Fill P matrix
        P[i, np.concatenate((np.arange(i), np.arange(i + 1, n)))] = Pi
        betas[i] = beta

    # Symmetrize P
    P = (P + P.T) / (2 * n)

    return P
