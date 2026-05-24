#!/usr/bin/env python3
"""Expectation Maximization for a Gaussian Mixture Model"""

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Performs the expectation maximization for a GMM."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None

    if type(k) is not int or k <= 0:
        return None, None, None, None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None, None

    if type(tol) is not float or tol < 0:
        return None, None, None, None, None

    if type(verbose) is not bool:
        return None, None, None, None, None

    pi, m, S = initialize(X, k)

    if pi is None or m is None or S is None:
        return None, None, None, None, None

    g, l = expectation(X, pi, m, S)

    if g is None or l is None:
        return None, None, None, None, None

    if verbose:
        print("Log Likelihood after 0 iterations: {:.5f}".format(l))

    for i in range(1, iterations + 1):
        pi, m, S = maximization(X, g)

        if pi is None or m is None or S is None:
            return None, None, None, None, None

        g, new_l = expectation(X, pi, m, S)

        if g is None or new_l is None:
            return None, None, None, None, None

        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {:.5f}".format(i, new_l))

        if abs(new_l - l) <= tol:
            l = new_l

            if verbose and i % 10 != 0:
                print("Log Likelihood after {} iterations: {:.5f}".format(i, l))

            return pi, m, S, g, l

        l = new_l

    if verbose and iterations % 10 != 0:
        print("Log Likelihood after {} iterations: {:.5f}".format(iterations, l))

    return pi, m, S, g, l
