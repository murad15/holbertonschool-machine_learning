#!/usr/bin/env python3
"""Module for DeepNeuralNetwork class with private attributes"""


import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """
        Initializes the deep neural network
        nx: number of input features
        layers: list representing the nodes in each layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            # Check if layer nodes are positive integers inside the loop
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            # Determine dimensions for weight matrices
            # Layer 1 uses nx; subsequent layers use the size of the previous layer
            n_prev = nx if i == 0 else layers[i - 1]
            n_curr = layers[i]

            # He et al. initialization
            self.__weights["W{}".format(i + 1)] = (
                np.random.randn(n_curr, n_prev) * np.sqrt(2 / n_prev)
            )
            # Bias initialization to 0
            self.__weights["b{}".format(i + 1)] = np.zeros((n_curr, 1))

    @property
    def L(self):
        """Getter for the number of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter for the intermediary values cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for the weights dictionary"""
        return self.__weights
