#!/usr/bin/env python3
"""Something that function does"""


import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """Something that function does"""

    network.save_weights(filename, save_format = save_format)

def load_weights(network, filename):
    """Something that function does"""

    network.load_weights(filename)
