#!/usr/bin/env python3
"""qwe qw e qwe qwe  rewr"""


def mat_mul(mat1, mat2):
    """ qwe qe qe qwe qe qw e wqeqw eweq"""

    #Nof columns in mat1 must equal number of rows in mat2
    if len(mat1[0]) != len(mat2):
        return None

    rows = len(mat1)
    cols = len(mat2[0])
    common = len(mat2)

    result = []

    for i in range(rows):
        row = []
        for j in range(cols):
            s = 0
            for k in range(common):
                s += mat1[i][k] * mat2[k][j]
            row.append(s)
        result.append(row)

    return result
