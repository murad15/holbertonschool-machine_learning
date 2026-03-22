#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the costwork with L2 regularization
    Arguments:
    cost -- original cost (without L2 reguization)
    lambtha -- regularization parameter
    weights -- dictionary containing weights
               expected keys: "W1", "b1", ..., "WL", "bL"
    L -- number of layers
    m -- number of training examples
    Returns:
    L2-regularized cost
    """

    L2_sum = 0
    # Sum of squared weights (exclude biases)
    for i in range(1, L + 1):
        W = weights["W" + str(i)]
        L2_sum += np.sum(np.square(W))
    # L2 regularization term
    L2_cost = (lambtha / (2 * m)) * L2_sum
    return cost + L2_cost
