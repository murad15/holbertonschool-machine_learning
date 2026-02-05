#!/usr/bin/env python3
"""Something that function does"""


def determinant(matrix):
    """Something that function does"""

    # Type check: must be list of lists
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Special case: 0x0 matrix
    if matrix == [[]]:
        return 1

    # Square matrix check
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base cases
    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive expansion (Laplace)
    det = 0
    for c in range(n):
        sub_matrix = [
            row[:c] + row[c + 1:]
            for row in matrix[1:]
        ]
        det += ((-1) ** c) * matrix[0][c] * determinant(sub_matrix)

    return det
