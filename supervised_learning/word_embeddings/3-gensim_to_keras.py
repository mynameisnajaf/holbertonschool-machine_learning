#!/usr/bin/env python3
"""A module that does the trick"""
import tensorflow.keras as keras


def gensim_to_keras(model):
    """
    Converts a trained gensim Word2Vec model to a trainable Keras
    Embedding layer.
    """
    weights = model.wv.vectors

    embedding = keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True
    )

    return embedding
