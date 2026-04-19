#!/usr/bin/env python3
"""Something that function does"""

from tensorflow import keras as K

def identity_block(A_prev, filters):
    """
    Builds an identity block.

    Parameters:
    - A_prev: input tensor
    - filters: tuple/list (F11, F3, F12)

    Returns:
    - activated output tensor
    """

    F11, F3, F12 = filters

    initializer = K.initializers.he_normal(seed=0)

    # Save shortcut
    shortcut = A_prev

    # First component: 1x1 Conv
    X = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation('relu')(X)

    # Second component: 3x3 Conv
    X = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(X)
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation('relu')(X)

    # Third component: 1x1 Conv
    X = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(X)
    X = K.layers.BatchNormalization(axis=-1)(X)

    # Add shortcut
    X = K.layers.Add()([X, shortcut])

    # Final activation
    X = K.layers.Activation('relu')(X)

    return X
