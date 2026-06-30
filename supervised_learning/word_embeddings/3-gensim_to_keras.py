#!/usr/bin/env python3
"""A module that does the trick"""


def gensim_to_keras(model):
    """A function that does the trick"""
    return model.wv.get_keras_embedding(train_embeddings=True)
