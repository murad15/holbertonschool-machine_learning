#!/usr/bin/env python3
"""WQefqwfqw qwfqfq ew few  wfrw ef ewf ettwe"""


def poly_derivative(poly):
    """This functions outputs derivative coefficients"""

    for i in poly:
        if isinstance(i, int) or isinstance(i, float):
            pass
        else:
            return None
    coefs = []
    for q in range(1, len(poly)):
        coefs.append(poly[q]*q)
    return coefs
