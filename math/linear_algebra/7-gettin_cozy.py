#!/usr/bin/env python3
"""Qqweqwe qwe qwe qwe qe qe qeqw"""


def cat_matrices2D(mat1, mat2, axis=0):
    """wqe qweqw eqe qw eqwe qw eqwe"""

    if axis == 0:
        # Same number of columns required
        if len(mat1[0]) != len(mat2[0]):
            return None

        return [row[:] for row in mat1] + [row[:] for row in mat2]

    # Axis 1 → horizontal concatenation (stack columns)
    elif axis == 1:
        # Same number of rows required
        if len(mat1) != len(mat2):
            return None

        result = []
        for i in range(len(mat1)):
            result.append(mat1[i][:] + mat2[i][:])

        return result

    # Invalid axis
    return None
