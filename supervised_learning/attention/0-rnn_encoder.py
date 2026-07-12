#!/usr/bin/env python3
"""RNN encoder module for machine translation."""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """Represents an RNN encoder for machine translation."""

    def __init__(self, vocab, embedding, units, batch):
        """
        Initialize the RNN encoder.

        Args:
            vocab: Size of the input vocabulary.
            embedding: Dimensionality of the embedding vectors.
            units: Number of hidden units in the GRU.
            batch: Batch size.
        """
        super().__init__()

        self.batch = batch
        self.units = units

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )

        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform"
        )

    def initialize_hidden_state(self):
        """
        Initialize the GRU hidden state.

        Returns:
            Tensor of zeros with shape (batch, units).
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Perform forward propagation through the encoder.

        Args:
            x: Tensor of shape (batch, input_seq_len).
            initial: Initial hidden state of shape (batch, units).

        Returns:
            outputs: Tensor of shape
                (batch, input_seq_len, units).
            hidden: Last hidden state of shape (batch, units).
        """
        embedded = self.embedding(x)

        outputs, hidden = self.gru(
            embedded,
            initial_state=initial
        )

        return outputs, hidden
