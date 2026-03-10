#!/usr/bin/env python3
"""Something that function does"""


import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Something that function does"""

    y_pred = network.predict(data, verbose=verbose)

    return y_pred
