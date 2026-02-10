#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


def marginal(x, n, P, Pr):
    """
    Calculates the marginal probability of observing x successes out of n trials.
    """

    # Validate n
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # Validate x
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    # Validate P
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # Validate Pr
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    # Validate values in P
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Validate values in Pr
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    # Validate Pr sums to 1
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Binomial coefficient
    coeff = np.math.factorial(n) / (
        np.math.factorial(x) * np.math.factorial(n - x)
    )

    # Likelihood
    likelihood = coeff * (P ** x) * ((1 - P) ** (n - x))

    # Marginal probability (sum of intersections)
    return np.sum(likelihood * Pr)
