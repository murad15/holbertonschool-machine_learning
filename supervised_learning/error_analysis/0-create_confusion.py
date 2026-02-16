#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


def create_confusion_matrix(labels, logits):
    """Something that function does"""

    true_cls = np.argmax(labels, axis=1)
    pred_cls = np.argmax(logits, axis=1)

    classes = labels.shape[1]
    confusion = np.zeros((classes, classes), dtype=int)

    for i in range(labels.shape[0]):   # iterats
        confusion[true_cls[i], pred_cls[i]] += 1

    return confusion
