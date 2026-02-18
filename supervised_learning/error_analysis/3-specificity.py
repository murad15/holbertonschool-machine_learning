#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


def specificity(confusion):
    """Something that function does"""
    z = 0
    c = []
    for i in range(confusion.shape[0]):
        numerator = np.sum(np.delete(np.delete(confusion,i,1),i,0))
        denominator = np.sum(np.delete(confusion,i,0))
        sens=numerator/denominator
        c.append(sens)
        z += 1
    c = np.array(c)
    return c
