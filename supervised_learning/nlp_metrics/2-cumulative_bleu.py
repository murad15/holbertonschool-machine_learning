#!/usr/bin/env python3
"""
Cumulative BLEU score
"""

import numpy as np


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence.

    Args:
        references: list of reference translations
        sentence: list containing the model proposed sentence
        n: size of the largest n-gram to use for evaluation

    Returns:
        The cumulative n-gram BLEU score
    """
    sent_len = len(sentence)

    # Brevity penalty
    ref_len = min(
        [len(ref) for ref in references],
        key=lambda x: (abs(x - sent_len), x)
    )

    if sent_len > ref_len:
        bp = 1
    else:
        bp = np.exp(1 - ref_len / sent_len)

    precisions = []

    for k in range(1, n + 1):
        # Sentence k-grams
        sent_ngrams = [
            tuple(sentence[i:i + k])
            for i in range(sent_len - k + 1)
        ]

        sent_counts = {}
        for gram in sent_ngrams:
            sent_counts[gram] = sent_counts.get(gram, 0) + 1

        clipped_count = 0

        for gram, count in sent_counts.items():
            max_ref_count = 0

            for ref in references:
                ref_ngrams = [
                    tuple(ref[i:i + k])
                    for i in range(len(ref) - k + 1)
                ]

                ref_count = ref_ngrams.count(gram)
                max_ref_count = max(max_ref_count, ref_count)

            clipped_count += min(count, max_ref_count)

        precision = clipped_count / len(sent_ngrams)
        precisions.append(precision)

    # If any precision is 0, cumulative BLEU becomes 0
    if min(precisions) == 0:
        return 0

    # Even weights: each n-gram score has weight 1 / n
    score = bp * np.exp(np.sum((1 / n) * np.log(precisions)))

    return score
