#!/usr/bin/env python3
"""Polynomial derivative module."""


def poly_integral(poly, C=0):
    """Compute the derivative of a polynomial represented"""

    # Validate input: must be a non-empty list of numbers (int/float)
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    for c in poly:
        # Reject non-numeric coefficients (also rejects bool, since bool is a subclass of int)
        if isinstance(c, bool) or not isinstance(c, (int, float)) or not isinstance(C, (int, float)):
            return None

    # Derivative of a constant (degree 0) is 0
    if len(poly) == 1:
        return [0]

    coefs=[0]
    for i in range(1, len(poly)+1):
        coefs.append(poly[i-1]*(1/i))
    return coefs
