#!/usr/bin/env python3
"""Something that function does"""


def cofactor(matrix):
    """Something that function does"""

    # Type checks
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError("matrix must be a list of lists")

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)

    # Square + non-empty check
    if n == 0 or not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Helper: determinant
    def determinant(mat):
        size = len(mat)

        if size == 1:
            return mat[0][0]

        if size == 2:
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]

        det = 0
        for col in range(size):
            submatrix = [
                row[:col] + row[col + 1:]
                for row in mat[1:]
            ]
            det += ((-1) ** col) * mat[0][col] * determinant(submatrix)
        return det

    # Build cofactor matrix
    cof_matrix = []
    for i in range(n):
        cof_row = []
        for j in range(n):
            minor = [
                row[:j] + row[j + 1:]
                for k, row in enumerate(matrix) if k != i
            ]
            cof_row.append(((-1) ** (i + j)) * determinant(minor))
        cof_matrix.append(cof_row)

    return cof_matrix
