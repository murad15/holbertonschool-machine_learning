#!/usr/bin/env python3
"""Something that function does"""


shuffle_data = __import__('2-shuffle_data').shuffle_data
import numpy as np


def create_mini_batches(X, Y, batch_size):
    """Something that function does"""

    X_s, Y_s = shuffle_data(X, Y)
    batchs = []
    for i in range(0, X.shape[0], batch_size):
        X_batch = X_s[i:i+batch_size]
        Y_batch = Y_s[i:i+batch_size]
        batchs.append((X_batch, Y_batch))

    return batchs
