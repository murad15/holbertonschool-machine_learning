#!/usr/bin/env python3
"""Something that function does"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """qweqwe qwe  wrtwet qe qwqw eqwe"""

    inputs = K.Input(shape=(nx,))
    x = inputs

    for i in range(len(layers)):
        x = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)

        if i != len(layers) - 1:
            x = K.layers.Dropout(rate=1 - keep_prob)(x)

    model = K.Model(inputs=inputs, outputs=x)
    return model
