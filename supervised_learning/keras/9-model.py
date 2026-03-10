#!/usr/bin/env python3
"""Something that function does"""


import tensorflow.keras as K


def save_model(network, filename):
    """Something that function does"""

    network.save(filename)


def load_model(filename):
    """Something that function does"""

    network = K.models.load_model(filename)
    return network
