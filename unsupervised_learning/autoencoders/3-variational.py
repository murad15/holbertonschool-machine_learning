#!/usr/bin/env python3
"""Variational autoencoder."""

import tensorflow.keras as keras
from tensorflow.keras import backend as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims: integer, dimensions of the model input
        hidden_layers: list containing the number of nodes for each hidden layer
        latent_dims: integer, dimensions of the latent space

    Returns:
        encoder, decoder, auto
    """

    def sampling(args):
        """Uses the reparameterization trick."""
        z_mean, z_log_var = args
        epsilon = K.random_normal(
            shape=(K.shape(z_mean)[0], latent_dims)
        )
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # Encoder
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input

    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation="relu")(x)

    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])

    encoder = keras.Model(
        inputs=encoder_input,
        outputs=[z, z_mean, z_log_var]
    )

    # Decoder
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation="relu")(x)

    decoder_output = keras.layers.Dense(
        input_dims,
        activation="sigmoid"
    )(x)

    decoder = keras.Model(
        inputs=decoder_input,
        outputs=decoder_output
    )

    # Full autoencoder
    auto_input = keras.Input(shape=(input_dims,))
    encoded, mean, log_var = encoder(auto_input)
    decoded = decoder(encoded)

    auto = keras.Model(
        inputs=auto_input,
        outputs=decoded
    )

    auto.compile(
        optimizer="adam",
        loss="binary_crossentropy"
    )

    return encoder, decoder, auto
