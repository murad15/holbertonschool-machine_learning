#!/usr/bin/env python3
"""Something that function does"""


import numpy as np

def likelihood(x, n, P):
    """
    Calculates the likelihood of observing x successes out of n trials
    for each hypothetical probability in P.
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

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Compute binomial coefficient
    coeff = np.math.factorial(n) / (
        np.math.factorial(x) * np.math.factorial(n - x)
    )

    # Compute likelihood for each probability in P
    return coeff * (P ** x) * ((1 - P) ** (n - x))

