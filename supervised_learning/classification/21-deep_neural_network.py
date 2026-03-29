#!/usr/bin/env python3
"""Module for DeepNeuralNetwork class with private attributes"""


import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Class constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError(
                    "layers must be a list of positive integers"
                )

            n_curr = layers[i]
            n_prev = nx if i == 0 else layers[i - 1]

            self.__weights["W{}".format(i + 1)] = (
                np.random.randn(n_curr, n_prev) *
                np.sqrt(2 / n_prev)
            )
            self.__weights["b{}".format(i + 1)] = np.zeros((n_curr, 1))

    @property
    def L(self):
        """Getter for number of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates forward propagation of the neural network"""
        self.__cache["A0"] = X

        for i in range(1, self.__L + 1):
            W = self.__weights["W{}".format(i)]
            b = self.__weights["b{}".format(i)]
            A_prev = self.__cache["A{}".format(i - 1)]

            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))

            self.__cache["A{}".format(i)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """Calculates the cost using logistic regression"""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) +
                       (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = (A >= 0.5).astype(int)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent"""
        m = Y.shape[1]
        weights = self.__weights.copy()
        dZ = cache["A{}".format(self.__L)] - Y

        for l in reversed(range(1, self.__L + 1)):
            A_prev = cache["A{}".format(l - 1)]
            W = weights["W{}".format(l)]

            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            self.__weights["W{}".format(l)] = W - alpha * dW
            self.__weights["b{}".format(l)] -= alpha * db

            if l > 1:
                A_prev = cache["A{}".format(l - 1)]
                dZ = np.matmul(W.T, dZ) * (A_prev * (1 - A_prev))
