#!/usr/bin/env python3
"""qwdqd qwq d qwd qd qwd qwd wqd"""

import numpy as np


class Neuron:
    """Class that defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """Initialize the neuron qwdq wq dasd asd"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Private attributes
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights dasdas ad as asd asd a"""
        return self.__W

    @property
    def b(self):
        """Getter for bias sdf sdf sdf sdfqw"""
        return self.__b

    @property
    def A(self):
        """Getter for activated output dqdwd qd """
        return self.__A

    def forward_prop(self, X):
        """Calculates dfs df sfs fdsfd"""
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Calculates cost of neural net"""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) +
                       (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
       """Calculates cost of neural net"""
        A = self.forward_prop(X)
        prediction = (A >= 0.5).astype(int)
        cost = self.cost(Y, A)
        return prediction, cost
