#!/usr/bin/env python3
"""Very interesting function omg"""


def summation_i_squared(n):
    """This function does something interesting"""

    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
