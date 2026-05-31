#!/usr/bin/env python3
"""Gaussian Process module."""

import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Initialize the Gaussian process.

        Args:
            X_init: numpy.ndarray of shape (t, 1)
            Y_init: numpy.ndarray of shape (t, 1)
            l: length parameter for the kernel
            sigma_f: output standard deviation
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculate the RBF covariance kernel matrix.

        Args:
            X1: numpy.ndarray of shape (m, 1)
            X2: numpy.ndarray of shape (n, 1)

        Returns:
            numpy.ndarray of shape (m, n)
        """
        sqdist = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + \
            np.sum(X2 ** 2, axis=1) - 2 * np.matmul(X1, X2.T)

        return (self.sigma_f ** 2) * np.exp(-0.5 * sqdist / (self.l ** 2))

    def predict(self, X_s):
        """
        Predict the mean and variance of points in the Gaussian process.

        Args:
            X_s: numpy.ndarray of shape (s, 1)

        Returns:
            mu: numpy.ndarray of shape (s,), predicted means
            sigma: numpy.ndarray of shape (s,), predicted variances
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu = K_s.T @ K_inv @ self.Y
        sigma = K_ss - K_s.T @ K_inv @ K_s

        return mu.reshape(-1), np.diag(sigma)
