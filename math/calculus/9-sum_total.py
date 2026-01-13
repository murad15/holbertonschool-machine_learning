#!/usr/bin/env python3
"""Very interesting function omg"""


def summation_i_squared(n):
    """This function does something interesting"""

    if n>=0 and isinstance(n,int):
        res = n*(n+1)*(2*n+1)/6
        return res
    else:
        return None
