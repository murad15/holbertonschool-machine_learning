#!/usr/bin/env python3
"""
Gensim to Keras
"""

import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a trained gensim Word2Vec model to a Keras Embedding layer.

    Args:
        model: trained gensim Word2Vec model

    Returns:
        trainable Keras Embedding layer
    """
    weights = model.wv.vectors

    embedding = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        embeddings_initializer=tf.keras.initializers.Constant(weights),
        trainable=True
    )

    return embedding
