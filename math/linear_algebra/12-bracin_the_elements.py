#!/usr/bin/env python3
"""Something that function does"""


import numpy as np

def np_elementwise(mat1, mat2):
    """Something that function does"""

    mat1 = np.array(mat1)
    mat2 = np.array(mat2)
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2
