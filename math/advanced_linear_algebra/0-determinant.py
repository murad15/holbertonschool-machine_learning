#!/usr/bin/env python3
"""Something that function does"""


def determinant(matrix):
    """Something that function does"""

    if not isinstance(matrix, list) or not isinstance(matrix[0], list):
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        return 1
    try:
        det = numpy.linalg.det(matrix)
        return det
    except ValueError:
        print("matrix must be a square matrix")
