#!/usr/bin/env python3
"""WQefqwfqw qwfqfq ew few  wfrw ef ewf ettwe"""


def poly_derivative(poly):
    """This functions outputs derivative coefficients"""

    # Validate input: must be a non-empty list of numbers (int/float)
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    for c in poly:
        # Reject non-numeric coefficie
        if isinstance(c, bool) or not isinstance(c, (int, float)):
            return None

    # Derivative of a constant (degree 0) is 0
    if len(poly) == 1:
        return [0]

    # Build derivative: d/dx (c_i * x^i) = i * c_i * x^(i-1) for i >= 1
    deriv = [i * poly[i] for i in range(1, len(poly))]

    # If all coefficients are zero, derivative is zero polynomial
    if all(v == 0 for v in deriv):
        return [0]

    return deriv
