#!/usr/bin/env python3
"""Convolutional autoencoder model."""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder.

    Args:
        input_dims: tuple, dimensions of the model input
        filters: list, number of filters for each encoder convolutional layer
        latent_dims: tuple, dimensions of the latent space representation

    Returns:
        encoder, decoder, auto
    """

    # Encoder
    encoder_input = keras.Input(shape=input_dims)
    x = encoder_input

    for f in filters:
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            activation="relu",
            padding="same"
        )(x)
        x = keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            padding="same"
        )(x)

    encoder = keras.Model(inputs=encoder_input, outputs=x)

    # Decoder
    decoder_input = keras.Input(shape=latent_dims)
    x = decoder_input

    for f in reversed(filters[1:]):
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            activation="relu",
            padding="same"
        )(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)

    # Second-to-last convolution: valid padding
    x = keras.layers.Conv2D(
        filters=filters[0],
        kernel_size=(3, 3),
        activation="relu",
        padding="valid"
    )(x)
    x = keras.layers.UpSampling2D(size=(2, 2))(x)

    # Last convolution: output channels = input channels
    decoder_output = keras.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        activation="sigmoid",
        padding="same"
    )(x)

    decoder = keras.Model(inputs=decoder_input, outputs=decoder_output)

    # Full autoencoder
    auto_input = keras.Input(shape=input_dims)
    encoded = encoder(auto_input)
    decoded = decoder(encoded)

    auto = keras.Model(inputs=auto_input, outputs=decoded)

    auto.compile(
        optimizer="adam",
        loss="binary_crossentropy"
    )

    return encoder, decoder, auto
