#!/usr/bin/env python3
"""Deep Neural Network for multiclass classification"""

import numpy as np


class DeepNeuralNetwork:
    """Deep Neural Network performing multiclass classification"""

    def __init__(self, nx, layers):
        """Class constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        for nodes in layers:
            if not isinstance(nodes, int) or nodes < 1:
                raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):

            if i == 0:
                he = np.sqrt(2 / nx)
                self.__weights["W1"] = np.random.randn(layers[i], nx) * he
            else:
                he = np.sqrt(2 / layers[i - 1])
                self.__weights["W{}".format(i + 1)] = (
                    np.random.randn(layers[i], layers[i - 1]) * he
                )

            self.__weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Number of layers"""
        return self.__L

    @property
    def cache(self):
        """Cache dictionary"""
        return self.__cache

    @property
    def weights(self):
        """Weights dictionary"""
        return self.__weights

    def softmax(self, Z):
        """Softmax activation"""
        exp = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp / np.sum(exp, axis=0, keepdims=True)

    def forward_prop(self, X):
        """Forward propagation"""
        self.__cache["A0"] = X

        for i in range(1, self.__L + 1):

            W = self.__weights["W{}".format(i)]
            b = self.__weights["b{}".format(i)]
            A_prev = self.__cache["A{}".format(i - 1)]

            Z = np.matmul(W, A_prev) + b

            if i == self.__L:
                A = self.softmax(Z)
            else:
                A = 1 / (1 + np.exp(-Z))

            self.__cache["A{}".format(i)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """Cross-entropy cost for multiclass"""
        m = Y.shape[1]

        cost = -np.sum(Y * np.log(A)) / m

        return cost

    def evaluate(self, X, Y):
        """Evaluate network predictions"""
        A, _ = self.forward_prop(X)

        predictions = np.zeros_like(A)
        predictions[np.argmax(A, axis=0), np.arange(A.shape[1])] = 1

        cost = self.cost(Y, A)

        return predictions, cost
