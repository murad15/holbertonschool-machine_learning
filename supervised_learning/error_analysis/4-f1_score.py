#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


def f1_score(confusion):
    """Something that function does"""

    sensitivity = __import__('1-sensitivity').sensitivity
    precision = __import__('2-precision').precision

    f1 = (2*sensitivity(confusion)*precision(confusion))/(precision(confusion)+sensitivity(confusion))
    return f1
