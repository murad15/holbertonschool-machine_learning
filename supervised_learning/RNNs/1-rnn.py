#!/usr/bin/env python3
"""
Forward propagation for a simple RNN
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN.

    Args:
        rnn_cell: instance of RNNCell
        X: input data, shape (t, m, i)
        h_0: initial hidden state, shape (m, h)

    Returns:
        H: all hidden states, shape (t + 1, m, h)
        Y: all outputs, shape (t, m, o)
    """
    t = X.shape[0]
    m = X.shape[1]
    h = h_0.shape[1]
    o = rnn_cell.by.shape[1]

    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))

    H[0] = h_0

    h_next = h_0

    for step in range(t):
        h_next, y = rnn_cell.forward(h_next, X[step])
        H[step + 1] = h_next
        Y[step] = y

    return H, Y
