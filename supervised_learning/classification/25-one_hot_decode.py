#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels.

    Parameters:
    one_hot (numpy.ndarray): one-hot encoded array with shape (classes, m)

    Returns:
    numpy.ndarray: shape (m,) containing numeric labels, or None on failure
    """

    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None

    try:
        labels = np.argmax(one_hot, axis=0)
        return labels
    except Exception:
        return None
