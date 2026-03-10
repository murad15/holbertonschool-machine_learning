#!/usr/bin/env python3
"""Something that function does"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Qweqe qweqw eqweqw dasdsad """

    regularizer = K.regularizers.l2(lambtha)

    x = K.layers.Dense(
        layers[0],
        activation=activations[0],
        kernel_regularizer=regularizer,
        input_shape=(nx,)
    )

    inputs = x.input
    outputs = x.output

    for i in range(1, len(layers)):
        outputs = K.layers.Dropout(1 - keep_prob)(outputs)
        outputs = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=regularizer
        )(outputs)

    model = K.Model(inputs=inputs, outputs=outputs)
    return model
