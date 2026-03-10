#!/usr/bin/env python3
"""Something that function does"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Qweqe qweqw eqweqw dasdsad """

    model = K.Sequential()
    regularizer = K.regularizers.l2(lambtha)

    for i in range(len(layers)):
        if i == 0:
            model.add(
                K.layers.Dense(
                    units=layers[i],
                    activation=activations[i],
                    kernel_regularizer=regularizer,
                    input_shape=(nx,)
                )
            )
        else:
            model.add(K.layers.Dropout(rate=1 - keep_prob))
            model.add(
                K.layers.Dense(
                    units=layers[i],
                    activation=activations[i],
                    kernel_regularizer=regularizer
                )
            )

    return model
