#!/usr/bin/env python3
"""Calculate Shannon entropy and P affinities"""

import numpy as np


def HP(Di, beta):
    """
    Calculates the Shannon entropy and P affinities
    relative to a data point.

    Parameters:
    Di : numpy.ndarray of shape (n - 1,)
        Pairwise distances between a point and all other points
    beta : numpy.ndarray of shape (1,)
        Beta value for the Gaussian distribution

    Returns:
    Hi : float
        Shannon entropy of the points
    Pi : numpy.ndarray of shape (n - 1,)
        P affinities of the points
    """

    # Compute affinities
    Pi = np.exp(-Di * beta)

    # Normalize affinities
    sumPi = np.sum(Pi)
    Pi = Pi / sumPi

    # Compute Shannon entropy
    Hi = -np.sum(Pi * np.log2(Pi))

    return Hi, Pi
