#!/usr/bin/env python3
"""Something that function does"""


import tensorflow.keras as K


def save_config(network, filename):
    """Something that function does"""

    with open(filename, 'w') as f:
        f.write(network.to_json())

def load_config(network, filename):
    """Something that function does"""

    with open(filename, 'r') as f:
        json_config = f.read()

    model = K.models.model_from_json(json_config)
    return model
