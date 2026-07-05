#!/usr/bin/env python3
"""
LSTM Cell
"""

import numpy as np


class LSTMCell:
    """
    Represents an LSTM unit.
    """

    def __init__(self, i, h, o):
        """
        Initializes the LSTM cell.

        Args:
            i: dimensionality of the input data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step.

        Args:
            h_prev: previous hidden state, shape (m, h)
            c_prev: previous cell state, shape (m, h)
            x_t: input data at time step t, shape (m, i)

        Returns:
            h_next: next hidden state
            c_next: next cell state
            y: output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        f = self.sigmoid(np.matmul(concat, self.Wf) + self.bf)
        u = self.sigmoid(np.matmul(concat, self.Wu) + self.bu)
        c_inter = np.tanh(np.matmul(concat, self.Wc) + self.bc)
        o = self.sigmoid(np.matmul(concat, self.Wo) + self.bo)

        c_next = f * c_prev + u * c_inter
        h_next = o * np.tanh(c_next)

        y_linear = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax(y_linear)

        return h_next, c_next, y

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
