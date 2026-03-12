#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


def shuffle_data(X, Y):
    """Something that function does"""

    X_shuf = np.random.permutation(X)
    Y_shuf = np.random.permutation(Y)

    return X_shuf, Y_shuf
