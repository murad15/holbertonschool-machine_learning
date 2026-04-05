#!/usr/bin/env python3
"""Builds a modified LeNet-5 architecture using keras"""

from tensorflow import keras as K


def lenet5(X):
    """Builds the modified LeNet-5 model"""

    init = K.initializers.he_normal(seed=0)

    # Layer 1: Conv
    x = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(X)

    # Layer 2: MaxPool
    x = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(x)

    # Layer 3: Conv
    x = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=init
    )(x)

    # Layer 4: MaxPool
    x = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(x)

    # Flatten
    x = K.layers.Flatten()(x)

    # FC1
    x = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=init
    )(x)

    # FC2
    x = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=init
    )(x)

    # Output
    outputs = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=init
    )(x)

    model = K.models.Model(inputs=X, outputs=outputs)

    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
