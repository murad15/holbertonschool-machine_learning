#!/usr/bin/env python3
"""Something that function does"""


import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Something that function does"""

    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # Dense layer (without activation yet)
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=initializer,
        use_bias=False
    )(prev)

    # Batch normalization layer
    batch_norm = tf.keras.layers.BatchNormalization(
        epsilon=1e-7,
        gamma_initializer=tf.keras.initializers.Ones(),
        beta_initializer=tf.keras.initializers.Zeros()
    )(dense)

    # Apply activation
    output = activation(batch_norm)

    return output
