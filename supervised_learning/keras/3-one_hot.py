#!/usr/bin/env python3
"""Something that function does"""


import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Something that function does"""

    onehot_labels = K.ops.one_hot(
        labels, num_classes=classes, axis=-1
    )

    return onehot_labels
