#!/usr/bin/env python3
"""Dataset class for Portuguese-to-English machine translation."""

from transformers import AutoTokenizer

# Adjust this import based on the filename containing load_pt2en
load_pt2en = __import__("0-dataset").load_pt2en


class Dataset:
    """Loads and prepares the Portuguese-to-English TED dataset."""

    def __init__(self):
        """Initialize the training data, validation data, and tokenizers."""
        self.data_train = load_pt2en("train")
        self.data_valid = load_pt2en("validation")

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Create Portuguese and English subword tokenizers.

        Args:
            data: tf.data.Dataset containing (Portuguese, English) pairs.

        Returns:
            tokenizer_pt: Trained Portuguese tokenizer.
            tokenizer_en: Trained English tokenizer.
        """
        base_tokenizer_pt = AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )
        base_tokenizer_en = AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        def portuguese_iterator():
            for pt, _ in data:
                yield pt.numpy().decode("utf-8")

        def english_iterator():
            for _, en in data:
                yield en.numpy().decode("utf-8")

        vocab_size = 2 ** 13

        tokenizer_pt = base_tokenizer_pt.train_new_from_iterator(
            portuguese_iterator(),
            vocab_size=vocab_size,
        )

        tokenizer_en = base_tokenizer_en.train_new_from_iterator(
            english_iterator(),
            vocab_size=vocab_size,
        )

        return tokenizer_pt, tokenizer_en
