#!/usr/bin/env python3
"""Something that function does"""


def minor(matrix):
    # Check type: must be list of lists
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check non-empty square matrix
    if (
        len(matrix) == 0
        or any(len(row) != len(matrix) for row in matrix)
    ):
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)

    # Special case: 1x1 matrix
    if n == 1:
        return [[1]]

    def determinant(mat):
        if len(mat) == 1:
            return mat[0][0]
        if len(mat) == 2:
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]

        det = 0
        for c in range(len(mat)):
            sub = [
                row[:c] + row[c+1:]
                for row in mat[1:]
            ]
            det += ((-1) ** c) * mat[0][c] * determinant(sub)
        return det

    # Build minor matrix
    minor_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            sub_matrix = [
                matrix[r][:j] + matrix[r][j+1:]
                for r in range(n) if r != i
            ]
            row.append(determinant(sub_matrix))
        minor_matrix.append(row)

    return minor_matrix
