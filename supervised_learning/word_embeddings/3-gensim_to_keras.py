#!/usr/bin/env python3
"""
Gensim to Keras
"""

from tensorflow import keras


def gensim_to_keras(model):
    """
    Converts a trained gensim Word2Vec model to a Keras Embedding layer.

    Args:
        model: trained gensim Word2Vec model

    Returns:
        trainable Keras Embedding layer
    """
    weights = model.wv.vectors

    embedding = keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True
    )

    return embedding
