#!/usr/bin/env python3
"""Something that function does"""


import tensorflow.keras as K


def save_config(network, filename):
    """Something that function does"""

    network.to_json(filename)


def load_config(network, filename):
    """Something that function does"""

    network = K.models.model_from_json(filename)
    return network
