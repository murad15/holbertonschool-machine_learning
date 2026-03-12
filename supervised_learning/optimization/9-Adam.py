#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Something that function does"""

    # 1. Update the first moment (momentum)
    v_new = beta1 * v + (1 - beta1) * grad

    # 2. Update the second moment (RMSProp-style)
    s_new = beta2 * s + (1 - beta2) * (grad**2)

    # 3. Bias correction for the first moment
    v_corrected = v_new / (1 - (beta1**t))

    # 4. Bias correction for the second moment
    s_corrected = s_new / (1 - (beta2**t))

    # 5. Update the variable in place
    # var[:] ensures the arraer than reassigned
    var[:] = var - alpha * (v_corrected / (np.sqrt(s_corrected) + epsilon))

    return var, v_new, s_new
