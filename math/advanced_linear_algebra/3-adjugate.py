#!/usr/bin/env python3
"""Something that function does"""


def adjugate(matrix):
    """Something that function does"""

    if not isinstance(matrix, list) or matrix == []:
        raise TypeError("matrix must be a list of lists")

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)

    # Square + non-empty check
    if n == 0 or not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # 1x1 special case
    if n == 1:
        return [[1]]

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

    # Step 1: Cofactor matrix
    cofactor = []
    for i in range(n):
        row = []
        for j in range(n):
            minor = [
                r[:j] + r[j + 1:]
                for k, r in enumerate(matrix) if k != i
            ]
            row.append(((-1) ** (i + j)) * determinant(minor))
        cofactor.append(row)

    # Step 2: Transpose (adjugate)
    adj = []
    for j in range(n):
        adj_row = []
        for i in range(n):
            adj_row.append(cofactor[i][j])
        adj.append(adj_row)

    return adj
