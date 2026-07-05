#!/usr/bin/env python3
"""
N-gram BLEU score
"""

import numpy as np


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence.

    Args:
        references: list of reference translations
        sentence: list containing the model proposed sentence
        n: size of the n-gram to use

    Returns:
        The n-gram BLEU score
    """
    sent_len = len(sentence)

    # Create n-grams from proposed sentence
    sentence_ngrams = []
    for i in range(sent_len - n + 1):
        sentence_ngrams.append(tuple(sentence[i:i + n]))

    # Count proposed sentence n-grams
    sentence_counts = {}
    for ngram in sentence_ngrams:
        sentence_counts[ngram] = sentence_counts.get(ngram, 0) + 1

    # Clipped n-gram precision
    clipped_count = 0

    for ngram, count in sentence_counts.items():
        max_ref_count = 0

        for reference in references:
            ref_ngrams = []
            for i in range(len(reference) - n + 1):
                ref_ngrams.append(tuple(reference[i:i + n]))

            ref_count = ref_ngrams.count(ngram)
            max_ref_count = max(max_ref_count, ref_count)

        clipped_count += min(count, max_ref_count)

    precision = clipped_count / len(sentence_ngrams)

    # Closest reference length
    closest_ref_len = min(
        [len(ref) for ref in references],
        key=lambda ref_len: (abs(ref_len - sent_len), ref_len)
    )

    # Brevity penalty
    if sent_len > closest_ref_len:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - closest_ref_len / sent_len)

    return brevity_penalty * precision
