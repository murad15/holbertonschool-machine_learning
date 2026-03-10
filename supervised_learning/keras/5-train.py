#!/usr/bin/env python3
"""Something that function does"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, verbose=True, shuffle=False):
    """Something that function does"""

    if not validation_data:

        history = network.fit(
            x=data,
            y=labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            shuffle=shuffle
        )
    else:
        history = network.fit(
            x=data,
            y=labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            shuffle=shuffle,
            validation_data=validation_data
        )

    return history
