#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


def softmax(Z):
    """Compute softmax"""
    exp = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp / np.sum(exp, axis=0, keepdims=True)

def dropout_forward_prop(X, weights, L, keep_prob):
    """Forward propagation with Dropout"""

    cache = {}
    cache["A0"] = X

    for i in range(1, L + 1):
        W = weights["W" + str(i)]
        b = weights["b" + str(i)]
        A_prev = cache["A" + str(i - 1)]

        Z = np.matmul(W, A_prev) + b
        if i == L:
            A = softmax(Z)
        else:
            A = np.tanh(Z)

            D = np.random.rand(*A.shape) < keep_prob
            A = (A * D) / keep_prob
            cache["D" + str(i)] = D

        cache["A" + str(i)] = A
    return cache
