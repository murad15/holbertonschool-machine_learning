#!/usr/bin/env python3
"""Something that function does"""


import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Something that function does"""

    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # Dense layer
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=initializer
    )(prev)

    # Trainable parameters
    gamma = tf.Variable(tf.ones([n]))
    beta = tf.Variable(tf.zeros([n]))

    epsilon = 1e-7

    # Compute batch statistics
    mean, variance = tf.nn.moments(dense, axes=[0])

    # Normalize
    normalized = (dense - mean) / tf.sqrt(variance + epsilon)

    # Scale and shift
    batch_norm = gamma * normalized + beta

    # Apply activation
    return activation(batch_norm)
