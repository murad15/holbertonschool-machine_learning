#!/usr/bin/env python3
"""Something that function does"""


import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Something that function does"""

    loss, accuracy = network.evaluate(x=data, y=labels, verbose=verbose)

    return [loss, accuracy]
