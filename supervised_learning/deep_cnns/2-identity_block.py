#!/usr/bin/env python3
"""Something that function does"""


from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Something that function does"""

    F11, F3, F12 = filters
    initializer = K.initializers.he_normal(seed=0)

    shortcut = A_prev

    # 1x1
    X = K.layers.Conv2D(F11, (1, 1), padding='same',
                        kernel_initializer=initializer)(A_prev)
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.ReLU()(X)

    # 3x3
    X = K.layers.Conv2D(F3, (3, 3), padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.ReLU()(X)

    # 1x1
    X = K.layers.Conv2D(F12, (1, 1), padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=-1)(X)

    # Add
    X = K.layers.Add()([X, shortcut])
    X = K.layers.ReLU()

    return X
