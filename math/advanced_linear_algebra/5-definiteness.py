#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


def definiteness(matrix):
    """Something that function does"""

    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Must be a valid square matrix
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0:
        return None

    # Must be symmetric
    if not np.allclose(matrix, matrix.T):
        return None

    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(matrix)

    # Tolerance for floating point comparisons
    tol = 1e-10

    positive = eigenvalues > tol
    negative = eigenvalues < -tol
    zero = np.abs(eigenvalues) <= tol

    if np.all(positive):
        return "Positive definite"
    if np.all(positive | zero) and np.any(zero):
        return "Positive semi-definite"
    if np.all(negative):
        return "Negative definite"
    if np.all(negative | zero) and np.any(zero):
        return "Negative semi-definite"
    if np.any(positive) and np.any(negative):
        return "Indefinite"

    return None
