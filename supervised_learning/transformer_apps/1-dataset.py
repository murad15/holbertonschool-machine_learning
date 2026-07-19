#!/usr/bin/env python3
"""Dataset class for Portuguese-to-English machine translation."""

import transformers
from setup import load_pt2en


class Dataset:
    """Loads and prepares the Portuguese-to-English dataset."""

    def __init__(self):
        """Initialize datasets and tokenizers."""
        self.data_train = load_pt2en("train")
        self.data_valid = load_pt2en("validation")

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Create Portuguese and English subword tokenizers."""
        base_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )
        base_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        def pt_iterator():
            for pt, _ in data:
                yield pt.numpy().decode("utf-8")

        def en_iterator():
            for _, en in data:
                yield en.numpy().decode("utf-8")

        tokenizer_pt = base_pt.train_new_from_iterator(
            pt_iterator(),
            vocab_size=2 ** 13,
        )
        tokenizer_en = base_en.train_new_from_iterator(
            en_iterator(),
            vocab_size=2 ** 13,
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encode a Portuguese-English translation pair.

        The start token has index vocab_size and the end token has
        index vocab_size + 1.
        """
        pt_sentence = pt.numpy().decode("utf-8")
        en_sentence = en.numpy().decode("utf-8")

        pt_vocab_size = self.tokenizer_pt.vocab_size
        en_vocab_size = self.tokenizer_en.vocab_size

        pt_tokens = self.tokenizer_pt.encode(
            pt_sentence,
            add_special_tokens=False,
        )
        en_tokens = self.tokenizer_en.encode(
            en_sentence,
            add_special_tokens=False,
        )

        pt_tokens = (
            [pt_vocab_size]
            + pt_tokens
            + [pt_vocab_size + 1]
        )
        en_tokens = (
            [en_vocab_size]
            + en_tokens
            + [en_vocab_size + 1]
        )

        return pt_tokens, en_tokens
