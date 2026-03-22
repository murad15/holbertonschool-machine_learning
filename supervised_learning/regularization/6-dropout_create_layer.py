#!/usr/bin/env python3
"""Something that function does"""


import tensorflow as tf


def dropout_create_layer(prev, n, activation,
                         keep_prob, training=True):
    """Create layer with dropout"""

    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_avg')

    dense = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )

    A = dense(prev)

    dropout = tf.keras.layers.Dropout(rate=1 - keep_prob)

    return dropout(A, training=training)
