#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


def create_confusion_matrix(labels, logits):
    """Something that function does"""

    labs = np.argmax(labels, axis=1)
    logs = np.argmax(logits, axis=1)

    matrix = np.zeros(shape=(labels.shape[1], labels.shape[1]))
    for i in labels:
        matrix[labs[i],logs[i]] += 1
    return matrix
