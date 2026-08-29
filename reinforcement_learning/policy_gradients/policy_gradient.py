#!/usr/bin/env python3
"""Policy function."""

import numpy as np


def policy(matrix, weight):
    """Compute the policy using a matrix and its weights."""
    scores = np.matmul(matrix, weight)
    probabilities = np.exp(scores)

    return probabilities / np.sum(probabilities, axis=1, keepdims=True)
