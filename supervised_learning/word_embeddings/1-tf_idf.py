#!/usr/bin/env python3
"""
TF_IDF embedding representation
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer



def tf_idf(sentences, vocab=None):
    """
    Creates a TF_IDF embedding matrix.

    Args:
        sentences: list of sentences to analyze
        vocab: list of vocabulary words to use

    Returns:
        embeddings: numpy.ndarray of shape (s, f)
        features: numpy.ndarray of features used
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)

    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names_out()

    return embeddings, features
