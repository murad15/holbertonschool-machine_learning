#!/usr/bin/env python3
"""Something that function does"""

from tensorflow import keras as K

def inception_block(A_prev, filters):
    """Something that function does"""

        F1, F3R, F3, F5R, F5, FPP = filters

    # 1x1 convolution branch
    conv_1x1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)

    # 1x1 -> 3x3 branch
    conv_3x3 = K.layers.Conv2D(
        filters=F3R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)

    conv_3x3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )(conv_3x3)

    # 1x1 -> 5x5 branch
    conv_5x5 = K.layers.Conv2D(
        filters=F5R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)

    conv_5x5 = K.layers.Conv2D(
        filters=F5,
        kernel_size=(5, 5),
        padding='same',
        activation='relu'
    )(conv_5x5)

    # MaxPool -> 1x1 branch
    pool_proj = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding='same'
    )(A_prev)

    pool_proj = K.layers.Conv2D(
        filters=FPP,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(pool_proj)

    # Concatenate all branches
    output = K.layers.Concatenate(axis=-1)([
        conv_1x1,
        conv_3x3,
        conv_5x5,
        pool_proj
    ])

    return output

    
