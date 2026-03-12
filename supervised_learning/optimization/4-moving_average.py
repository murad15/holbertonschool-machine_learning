#!/usr/bin/env python3
"""Something that function does"""


def moving_average(data, beta):
    """Something that function does"""

    averages = []
    v = 0

    for i, x in enumerate(data, 1):
        v = beta * v + (1 - beta) * x
        v_corrected = v / (1 - beta ** i)
        averages.append(v_corrected)

    return averages
