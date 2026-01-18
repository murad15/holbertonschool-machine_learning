#!/usr/bin/env python3
"""Polynomial derivative module."""


def poly_integral(poly, C=0):
    """Compute the derivative of a polynomial represented"""

    # Validate input: must be a non-empty list of numbers (int/float)
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    for c in poly:
        # Reject non-numeric coefficients (and bools)
        if isinstance(c, bool) or not isinstance(c, (int, float)):
            return None

    # Validate C (must be an integer, not bool)
    if isinstance(C, bool) or not isinstance(C, int):
        return None

    # Build integral coefficients
    integ = [C]
    for i, c in enumerate(poly):
        value = c / (i + 1)

        # If result is a whole number, represent it as int
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        integ.append(value)

    # Make the list as small as possible: trim trailing zeros
    while len(integ) > 1 and integ[-1] == 0:
        integ.pop()

    return integ
