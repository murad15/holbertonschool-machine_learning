#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Something that function does"""

    # 1. Calculate the mean along the batch (m data points)
    # axis=0 calculates the mean for each feature (n)
    mu = np.mean(Z, axis=0)

    # 2. Calculate the variance along the batch
    sigma_squared = np.var(Z, axis=0)

    # 3. Normalize:n and divide by standard deviation
    # Z_hat = (Z - mu) / sqrt(sigma^2 + epsilon)
    Z_hat = (Z - mu) / np.sqrt(sigma_squared + epsilon)

    # 4. Scale and shift (Gamma and Beta)
    Z_norm = gamma * Z_hat + beta

    return Z_norm
