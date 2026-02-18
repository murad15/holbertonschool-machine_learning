#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


def sensitivity(confusion):
    """Something that function does"""
    z = 0
    c=[]
    for i in range(confusion.shape[0]):
        sens = float(confusion[i][z]/np.sum(confusion[i,:]))
        c.append(sens)
        z += 1
    c = np.array(c)
    return c
