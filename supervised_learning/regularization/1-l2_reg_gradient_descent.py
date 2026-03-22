#!/usr/bin/env python3
"""Something that function does"""

import numpy as np

def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates w and b using gradient with L2"""
    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y

    for i in reversed(range(1, L + 1)):
        A_prev = cache["A" + str(i - 1)]
        W = weights["W" + str(l)]

        # Gradients with L2 regularization
        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            A_prev_activation = cache["A" + str(i - 1)]
            r = (1 - np.power(A_prev_activation, 2))
            dZ = np.matmul(W.T, dZ) * r

        # Update weights and biases in place
        weights["W" + str(i)] -= alpha * dW
        weights["b" + str(i)] -= alpha * db
