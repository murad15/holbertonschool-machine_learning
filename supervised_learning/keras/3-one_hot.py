#!/usr/bin/env python3
"""Something that function does"""


import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Something that function does"""

    return K.utils.to_categorical(labels, num_classes=classes)
