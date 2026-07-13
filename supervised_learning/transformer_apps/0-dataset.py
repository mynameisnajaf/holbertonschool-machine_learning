#!/usr/bin/env python3
"""Dataset module"""

import transformers
from setup import load_pt2en


class Dataset:
    """Dataset class"""

    def __init__(self):
        """Class constructor"""
        self.data_train = load_pt2en(split="train")
        self.data_valid = load_pt2en(split="validation")

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Creates tokenizers for Portuguese and English.
        """
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )

        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        return tokenizer_pt, tokenizer_en
