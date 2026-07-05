#!/usr/bin/env python3
"""
GRU Cell
"""

import numpy as np


class GRUCell:
    """
    Represents a gated recurrent unit cell.
    """

    def __init__(self, i, h, o):
        """
        Initializes the GRU cell.

        Args:
            i: dimensionality of the input data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.

        Args:
            h_prev: previous hidden state, shape (m, h)
            x_t: input data at time step t, shape (m, i)

        Returns:
            h_next: next hidden state
            y: output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        z = self.sigmoid(np.matmul(concat, self.Wz) + self.bz)
        r = self.sigmoid(np.matmul(concat, self.Wr) + self.br)

        r_h_prev = r * h_prev
        concat_reset = np.concatenate((r_h_prev, x_t), axis=1)

        h_intermediate = np.tanh(
            np.matmul(concat_reset, self.Wh) + self.bh
        )

        h_next = (1 - z) * h_prev + z * h_intermediate

        y_linear = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax(y_linear)

        return h_next, y

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """
        Softmax activation function.
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
