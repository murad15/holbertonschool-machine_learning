#!/usr/bin/env python3
"""
Unigram BLEU score
"""

import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence.
    """
    sentence_counts = {}
    for word in sentence:
        sentence_counts[word] = sentence_counts.get(word, 0) + 1

    clipped_count = 0

    for word, count in sentence_counts.items():
        max_ref_count = 0

        for reference in references:
            ref_count = reference.count(word)
            max_ref_count = max(max_ref_count, ref_count)

        clipped_count += min(count, max_ref_count)

    precision = clipped_count / len(sentence)

    sent_len = len(sentence)
    closest_ref_len = min(
        [len(ref) for ref in references],
        key=lambda ref_len: (abs(ref_len - sent_len), ref_len)
    )

    if sent_len > closest_ref_len:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - closest_ref_len / sent_len)

    return brevity_penalty * precision
