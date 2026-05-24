#!/usr/bin/env python3
"""Bayesian Information Criterion for GMM"""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using BIC."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None

    if type(kmin) is not int or kmin <= 0:
        return None, None, None, None

    if kmax is None:
        kmax = X.shape[0]

    if type(kmax) is not int or kmax <= 0 or kmax <= kmin:
        return None, None, None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None

    if type(tol) is not float or tol < 0:
        return None, None, None, None

    if type(verbose) is not bool:
        return None, None, None, None

    n, d = X.shape
    l = []
    b = []
    results = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        if pi is None:
            return None, None, None, None

        p = (k * d) + (k * d * (d + 1) / 2) + (k - 1)
        bic = (p * np.log(n)) - (2 * log_likelihood)

        l.append(log_likelihood)
        b.append(bic)
        results.append((pi, m, S))

    l = np.array(l)
    b = np.array(b)

    best_index = np.argmin(b)
    best_k = kmin + best_index
    best_result = results[best_index]

    return best_k, best_result, l, b
