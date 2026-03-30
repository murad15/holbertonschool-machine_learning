#!/usr/bin/env python3
"""Deep Neural Network for multiclass classification"""

import numpy as np


class DeepNeuralNetwork:
    """Deep Neural Network performing multiclass classification"""

    def __init__(self, nx, layers, activation='sig'):
        """Class constructor"""

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if (not isinstance(layers, list) or len(layers) == 0):
            raise TypeError("layers must be a list of positive integers")

        for nodes in layers:
            if not isinstance(nodes, int) or nodes < 1:
                raise TypeError("layers must be a list of positive integers")

        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        for i in range(self.__L):

            if i == 0:
                he = np.sqrt(2 / nx)
                self.__weights["W1"] = np.random.randn(layers[i], nx) * he
            else:
                he = np.sqrt(2 / layers[i - 1])
                self.__weights["W{}".format(i + 1)] = \
                    np.random.randn(layers[i], layers[i - 1]) * he

            self.__weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    @property
    def activation(self):
        return self.__activation

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def softmax(self, Z):
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
                if self.__activation == 'sig':
                    A = self.sigmoid(Z)
                else:
                    A = np.tanh(Z)

            self.__cache["A{}".format(i)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """Cross-entropy cost"""
        m = Y.shape[1]
        return -np.sum(Y * np.log(A)) / m

    def evaluate(self, X, Y):
        """Evaluate predictions"""

        A, _ = self.forward_prop(X)

        prediction = np.zeros_like(A)
        prediction[np.argmax(A, axis=0), np.arange(A.shape[1])] = 1

        cost = self.cost(Y, A)

        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Gradient descent"""

        m = Y.shape[1]
        weights_copy = self.__weights.copy()

        dZ = cache["A{}".format(self.__L)] - Y

        for i in reversed(range(1, self.__L + 1)):

            A_prev = cache["A{}".format(i - 1)]
            W = weights_copy["W{}".format(i)]

            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            self.__weights["W{}".format(i)] -= alpha * dW
            self.__weights["b{}".format(i)] -= alpha * db

            if i > 1:

                A_prev = cache["A{}".format(i - 1)]

                if self.__activation == 'sig':
                    g = A_prev * (1 - A_prev)

                else:
                    g = 1 - (A_prev ** 2)

                dZ = np.matmul(W.T, dZ) * g
