#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """
        Class constructor
        nx: number of input features
        layers: list representing the number of nodes in each layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(self.L):
            # Validate the element first to satisfy the order of exceptions
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            # Initialization logic
            n_curr = layers[i]
            n_prev = nx if i == 0 else layers[i - 1]
            # He initialization: W = randn * sqrt(2/n_prev)
            self.weights["W{}".format(i + 1)] = (
                np.random.randn(n_curr, n_prev) * np.sqrt(2 / n_prev)
            )
            self.weights["b{}".format(i + 1)] = np.zeros((n_curr, 1))
