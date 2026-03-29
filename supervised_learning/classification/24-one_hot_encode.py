#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.

    Parameters:
    Y (numpy.ndarray): shape (m,) containing numeric class labels
    classes (int): maximum number of classes

    Returns:
    numpy.ndarray: one-hot encoding with shape (classes, m), or None on failure
    """

    if not isinstance(Y, np.ndarray) or Y.ndim != 1:
        return None
    if not isinstance(classes, int) or classes <= 0:
        return None

    try:
        m = Y.shape[0]
        one_hot = np.zeros((classes, m))
        one_hot[Y, np.arange(m)] = 1
        return one_hot
    except Exception:
        return None
