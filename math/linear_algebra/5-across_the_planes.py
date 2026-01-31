#!/usr/bin/env python3
"""qwe wqeq weq weq we qe qew qw"""


def add_matrices2D(mat1, mat2):
    """qeqw wqe qwe qweqewq qeqwe qe"""

    if len(mat1) != len(mat2):
        return None

    result = []

    for i in range(len(mat1)):
        # Check same number of columns in each row
        if len(mat1[i]) != len(mat2[i]):
            return None

        row = []
        for j in range(len(mat1[i])):
            row.append(mat1[i][j] + mat2[i][j])

        result.append(row)

    return result
