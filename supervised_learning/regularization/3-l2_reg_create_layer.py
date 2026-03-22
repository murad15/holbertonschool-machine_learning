#!/usr/bin/env python3
"""Something that function does"""


import tensorflow as tf

def l2_reg_create_layer(prev, n, activation, lambtha):
    """Create layer with L2 regularization"""

    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_avg')

    regularizer = tf.keras.regularizers.L2(lambtha)
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer
    )

    return layer(prev)
