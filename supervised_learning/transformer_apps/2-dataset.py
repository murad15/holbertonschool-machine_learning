#!/usr/bin/env python3
"""Dataset class for Portuguese-to-English machine translation."""

import transformers
from setup import load_pt2en


tf = transformers.utils.import_utils.importlib.import_module("tensorflow")


class Dataset:
    """Loads and prepares the Portuguese-to-English dataset."""

    def __init__(self):
        """Initialize, tokenize, and encode the datasets."""
        data_train = load_pt2en("train")
        data_validate = load_pt2en("validation")

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            data_train
        )

        self.data_train = data_train.map(self.tf_encode)
        self.data_validate = data_validate.map(self.tf_encode)

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
        """Encode a Portuguese-English translation pair."""
        pt = pt.numpy().decode("utf-8")
        en = en.numpy().decode("utf-8")

        pt_vocab_size = self.tokenizer_pt.vocab_size
        en_vocab_size = self.tokenizer_en.vocab_size

        pt_tokens = self.tokenizer_pt.encode(
            pt,
            add_special_tokens=False,
        )
        en_tokens = self.tokenizer_en.encode(
            en,
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

    def tf_encode(self, pt, en):
        """Apply encode through a TensorFlow Python wrapper."""
        pt_tokens, en_tokens = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64],
        )

        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
