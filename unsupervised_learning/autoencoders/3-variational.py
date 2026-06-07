#!/usr/bin/env python3
"""Variational autoencoder model."""

import tensorflow.keras as keras
from tensorflow.keras import backend as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims: integer, dimensions of the model input
        hidden_layers: list of nodes for each hidden encoder layer
        latent_dims: integer, dimensions of the latent space

    Returns:
        encoder, decoder, auto
    """

    def sampling(args):
        """Reparameterization trick."""
        mean, log_var = args
        epsilon = K.random_normal(
            shape=(K.shape(mean)[0], latent_dims),
            mean=0.0,
            stddev=1.0
        )
        return mean + K.exp(log_var / 2) * epsilon

    # Encoder
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input

    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation="relu")(x)

    mean = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    latent = keras.layers.Lambda(sampling)([mean, log_var])

    encoder = keras.Model(
        inputs=encoder_input,
        outputs=[latent, mean, log_var]
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

    # Full VAE
    auto_input = keras.Input(shape=(input_dims,))
    encoded, z_mean, z_log_var = encoder(auto_input)
    decoded = decoder(encoded)

    auto = keras.Model(
        inputs=auto_input,
        outputs=decoded
    )

    # VAE loss = reconstruction loss + KL divergence
    def vae_loss(y_true, y_pred):
        reconstruction_loss = keras.losses.binary_crossentropy(
            y_true,
            y_pred
        )
        reconstruction_loss *= input_dims

        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        return K.mean(reconstruction_loss + kl_loss)

    auto.compile(
        optimizer="adam",
        loss=vae_loss
    )

    return encoder, decoder, auto
