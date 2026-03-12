#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Something that function does"""

    e_s = beta2 * s + (1 - beta2) * (grad ** 2)

    var_n = var - (alpha / (np.sqrt(e_s + epsilon)) * grad

    return var_n, e_s
