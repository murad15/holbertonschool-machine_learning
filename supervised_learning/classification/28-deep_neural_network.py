#!/usr/bin/env python3
"""Deep Neural Network for multiclass classification"""


import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """Defines a deep neural network"""

    def __init__(self, nx, layers, activation='sig'):
        """Class constructor"""

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        for i in layers:
            if not isinstance(i, int) or i <= 0:
                raise TypeError("layers must be a list of positive integers")

        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        for i in range(self.__L):

            if i == 0:
                prev = nx
            else:
                prev = layers[i - 1]

            self.__weights["W{}".format(i + 1)] = (
                np.random.randn(layers[i], prev) * np.sqrt(2 / prev)
            )

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

    def forward_prop(self, X):
        """Calculates forward propagation"""

        self.__cache["A0"] = X

        for i in range(1, self.__L + 1):

            W = self.__weights["W{}".format(i)]
            b = self.__weights["b{}".format(i)]
            A_prev = self.__cache["A{}".format(i - 1)]

            Z = np.matmul(W, A_prev) + b

            if i == self.__L:
                exp = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = exp / np.sum(exp, axis=0, keepdims=True)

            else:
                if self.__activation == 'sig':
                    A = 1 / (1 + np.exp(-Z))
                else:
                    A = np.tanh(Z)

            self.__cache["A{}".format(i)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """Calculates cost using multiclass cross-entropy"""

        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A + 1e-8)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluates network predictions"""

        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)

        prediction = np.zeros_like(A)
        prediction[np.argmax(A, axis=0), np.arange(A.shape[1])] = 1

        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Performs one pass of gradient descent"""

        m = Y.shape[1]
        weights = self.__weights.copy()

        dZ = cache["A{}".format(self.__L)] - Y

        for i in reversed(range(1, self.__L + 1)):

            A_prev = cache["A{}".format(i - 1)]
            W = weights["W{}".format(i)]

            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            self.__weights["W{}".format(i)] = W - alpha * dW
            self.__weights["b{}".format(i)] -= alpha * db

            if i > 1:

                A_prev = cache["A{}".format(i - 1)]

                if self.__activation == 'sig':
                    g = A_prev * (1 - A_prev)
                else:
                    g = 1 - A_prev ** 2

                dZ = np.matmul(W.T, dZ) * g

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the neural network"""

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps = []

        for i in range(iterations + 1):

            A, cache = self.forward_prop(X)

            if i % step == 0 or i == iterations:

                cost = self.cost(Y, A)

                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))

                if graph:
                    costs.append(cost)
                    steps.append(i)

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.plot(steps, costs, 'b')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file"""

        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork"""

        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
