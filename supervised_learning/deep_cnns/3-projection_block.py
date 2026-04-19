#!/usr/bin/env python3
"""Something that function does"""


from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """Something that function does"""
    
    F11, F3, F12 = filters
    initializer = K.initializers.he_normal(seed=0)

    shortcut = A_prev

    X = K.layers.Conv2D(F11, (1, 1), strides=(s, s),
                        padding='same',
                        kernel_initializer=initializer)(A_prev)
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(F3, (3, 3), padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(F12, (1, 1), padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=-1)(X)

    shortcut = K.layers.Conv2D(F12, (1, 1), strides=(s, s),
                              padding='same',
                              kernel_initializer=initializer)(shortcut)
    shortcut = K.layers.BatchNormalization(axis=-1)(shortcut)

    X = K.layers.Add()([X, shortcut])
    X = K.layers.Activation('relu')(X)

    return X
