#!/usr/bin/env python3
"""Something that function does"""


import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Something that function does"""

    opt = tf.keras.optimizers.Adam(
        learning_rate = alpha,
        beta_1 = beta1,
        beta_2 = beta2
    )
    network.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
