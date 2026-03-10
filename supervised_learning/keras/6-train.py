#!/usr/bin/env python3
"""Something that function does"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                verbose=True, shuffle=False):
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
        earlystop = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )

        history = network.fit(
            x=data,
            y=labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            shuffle=shuffle,
            callbacks=earlystop,
            validation_data=validation_data
        )

    return history
