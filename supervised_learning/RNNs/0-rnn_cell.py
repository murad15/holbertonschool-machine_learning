#!/usr/bin/env python3
"""
Simple RNN Cell
"""

import numpy as np


class RNNCell:
    """
    Represents a cell of a simple RNN.
    """

    def __init__(self, i, h, o):
        """
        Initializes the RNN cell.

        Args:
            i: dimensionality of the input data
            h: dimensionality of the hidden state
            o: dimensionality of the output
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
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

        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)

        z = np.matmul(h_next, self.Wy) + self.by
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        y = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return h_next, y
