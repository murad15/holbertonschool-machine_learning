#!/usr/bin/env python3
"""Something that function does"""


import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Something that function does"""

    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # 2. Create the Dense base layer
    # Note: use_bias=False is common with Batch Norm because 
    # acts as the bias, making a separate bias term redundant.
    dense_layer = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=init,
        use_bias=False
    )

    # Pass the previous layer output through the Dense layer
    z = dense_layer(prev)

    # 3. Create the Batch Normalization layer
    # gamma_init='onesbeta_init='zeros' are the defaults in Keras
    batch_norm = tf.keras.layers.BatchNormalization(
        epsilon=1e-7,
        beta_initializer="zeros",
        gamma_initializer="ones"
    )

    # Apply Batch Normalization to the linear output z
    z_norm = batch_norm(z)

    # 4. Apply the activation function
    # If activationit returns the normalized linear output
    if activation is None:
        return z_norm
    return activation(z_norm)
