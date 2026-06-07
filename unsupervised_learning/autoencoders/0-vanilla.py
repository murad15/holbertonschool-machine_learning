#!/usr/bin/env python3
"""Autoencoder model."""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder.

    Args:
        input_dims: integer, dimensions of the model input
        hidden_layers: list of number of nodes for each hidden encoder layer
        latent_dims: integer, dimensions of the latent space

    Returns:
        encoder, decoder, auto
    """

    # Encoder
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input

    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation="relu")(x)

    latent = keras.layers.Dense(latent_dims, activation="relu")(x)

    encoder = keras.Model(inputs=encoder_input, outputs=latent)

    # Decoder
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation="relu")(x)

    decoder_output = keras.layers.Dense(input_dims, activation="sigmoid")(x)

    decoder = keras.Model(inputs=decoder_input, outputs=decoder_output)

    # Full autoencoder
    auto_input = keras.Input(shape=(input_dims,))
    encoded = encoder(auto_input)
    decoded = decoder(encoded)

    auto = keras.Model(inputs=auto_input, outputs=decoded)

    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
