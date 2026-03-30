#!/usr/bin/env python3
"""Deep Neural Network"""


import numpy as np
import pickle


class DeepNeuralNetwork:
    """Defines a deep neural network"""

    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if (not isinstance(layers, list) or len(layers) == 0 or
                not all(isinstance(x, int) and x > 0 for x in layers)):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if i == 0:
                w = np.random.randn(layers[i], nx) * np.sqrt(2/nx)
            else:
                w = np.random.randn(layers[i], layers[i-1]) * \
                    np.sqrt(2/layers[i-1])

            self.__weights["W{}".format(i+1)] = w
            self.__weights["b{}".format(i+1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """Forward propagation"""
        self.__cache["A0"] = X

        for i in range(self.__L):
            W = self.__weights["W{}".format(i+1)]
            b = self.__weights["b{}".format(i+1)]
            A_prev = self.__cache["A{}".format(i)]

            Z = np.matmul(W, A_prev) + b
            A = 1/(1 + np.exp(-Z))

            self.__cache["A{}".format(i+1)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """Cost function"""
        m = Y.shape[1]
        return -np.sum(Y*np.log(A) + (1-Y)*np.log(1.0000001-A)) / m

    def evaluate(self, X, Y):
        """Evaluate predictions"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Gradient descent"""
        m = Y.shape[1]
        weights = self.__weights.copy()

        for i in reversed(range(self.__L)):
            A = cache["A{}".format(i+1)]
            A_prev = cache["A{}".format(i)]

            if i == self.__L-1:
                dZ = A - Y
            else:
                W_next = weights["W{}".format(i+2)]
                dZ = np.matmul(W_next.T, dZ) * (A*(1-A))

            dW = np.matmul(dZ, A_prev.T)/m
            db = np.sum(dZ, axis=1, keepdims=True)/m

            self.__weights["W{}".format(i+1)] -= alpha*dW
            self.__weights["b{}".format(i+1)] -= alpha*db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Train network"""

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations+1):
            A, cache = self.forward_prop(X)

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        return self.evaluate(X, Y)

    def save(self, filename):
        """Save model"""
        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Load model"""
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
