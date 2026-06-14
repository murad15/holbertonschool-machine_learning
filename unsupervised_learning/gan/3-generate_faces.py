#!/usr/bin/env python3
"""Builds convolutional generator and discriminator models."""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def convolutional_GenDiscr():
    """
    Build a convolutional generator and discriminator.

    The generator receives latent vectors of shape (16,) and outputs
    generated images of shape (16, 16, 1).

    The discriminator receives images of shape (16, 16, 1) and outputs
    one score.

    Returns:
        tuple: generator model and discriminator model.
    """

    def get_generator():
        """
        Build the generator model.

        Returns:
            keras.Model: Generator model.
        """
        model = keras.Sequential(name="generator")

        model.add(keras.layers.Input(shape=(16,)))
        model.add(keras.layers.Dense(128, activation="tanh"))
        model.add(keras.layers.Dense(4 * 4 * 32, activation="tanh"))
        model.add(keras.layers.Reshape((4, 4, 32)))

        model.add(keras.layers.UpSampling2D())
        model.add(keras.layers.Conv2D(
            64,
            kernel_size=3,
            padding="same",
            activation="tanh"
        ))

        model.add(keras.layers.UpSampling2D())
        model.add(keras.layers.Conv2D(
            32,
            kernel_size=3,
            padding="same",
            activation="tanh"
        ))

        model.add(keras.layers.Conv2D(
            1,
            kernel_size=3,
            padding="same",
            activation="tanh"
        ))

        return model

    def get_discriminator():
        """
        Build the discriminator model.

        Returns:
            keras.Model: Discriminator model.
        """
        model = keras.Sequential(name="discriminator")

        model.add(keras.layers.Input(shape=(16, 16, 1)))
        model.add(keras.layers.Conv2D(
            32,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="tanh"
        ))

        model.add(keras.layers.Conv2D(
            64,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="tanh"
        ))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation="tanh"))
        model.add(keras.layers.Dense(1, activation="tanh"))

        return model

    return get_generator(), get_discriminator()
