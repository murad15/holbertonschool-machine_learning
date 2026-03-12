#!/usr/bin/env python3
"""Something that function does"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Something that function does"""

    v1 = beta1 * v + (1 - beta1) * grad

    var_n = var - alpha * v1

    return var_n, v1
