#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Forward propagation using Dropout"""
    cache = {}
    cache['A0'] = X

    for i in range(1, L + 1):
        W = weights[f'W{i}']
        b = weights[f'b{i}']
        A_prev = cache[f'A{i-1}']

        # Linear Step
        Z = np.matmul(W, A_prev) + b

        if i < L:
            # Tanh activation for hidden layers
            A = np.tanh(Z)
            mask = (np.random.rand(A.shape[0], A.shape[1])
                      < keep_prob).astype(int)
            A = (A * mask) / keep_prob
            cache[f'D{i}'] = mask
            cache[f'A{i}'] = A
        else:
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            cache[f'A{i}'] = A

    return cache
