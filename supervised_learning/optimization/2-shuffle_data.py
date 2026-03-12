#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


def shuffle_data(X, Y):
    """Something that function does"""

    permutation = np.random.permutation(X.shape[0])
    return X[permutation], Y[permutation]
