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

        for l in range(self.L):
            if not isinstance(layers[l], int) or layers[l] <= 0:
                raise TypeError("layers must be a list of positive integers")

            # Determine input size for the current layer
            # Layer 1 (index 0) takes nx as input; others take layers[l-1]
            n_prev = nx if l == 0 else layers[l - 1]
            n_curr = layers[l]

            # He et al. initialization
            self.weights[f'W{l + 1}'] = np.random.randn(n_curr, n_prev) * np.sqrt(2 / n_prev)
            # Bias initialization to 0
            self.weights[f'b{l + 1}'] = np.zeros((n_curr, 1))
