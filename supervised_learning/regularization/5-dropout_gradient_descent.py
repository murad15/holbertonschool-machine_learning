#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha,
                             keep_prob, L):
    """Update weights using gradient descent with Dropout"""
    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y

    for i in reversed(range(1, L + 1)):
        A_prev = cache["A" + str(i - 1)]
        W = weights["W" + str(i)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            dA = np.matmul(W.T, dZ)
            D = cache["D" + str(i - 1)]
            dA = (dA * D) / keep_prob
            A_prev = cache["A" + str(i - 1)]
            dZ = dA * (1 - np.power(A_prev, 2))

        weights["W" + str(i)] -= alpha * dW
        weights["b" + str(i)] -= alpha * db
