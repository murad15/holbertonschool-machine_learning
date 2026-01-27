#!/usr/bin/env python3
"""q ewe wq qwe qw eqw e qwe qwe  qe"""


def matrix_shape(matrix):
    """qweqw eqw eq we qwe qw e qe"""

    dims = []
    while isinstance(matrix, list):
        dims.append(len(matrix))
        if len(matrix) == 0:
            break
        matrix = matrix[0]
    return dims
