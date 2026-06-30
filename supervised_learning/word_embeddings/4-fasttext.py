#!/usr/bin/env python3
"""FastText model"""

import gensim


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """Creates and trains a FastText model."""

    if cbow is True:
        cbow_flag = 0
    else:
        cbow_flag = 1
    model = gensim.models.FastText(sentences=sentences,
                     vector_size=vector_size,
                     min_count=min_count,
                     window=window,
                     negative=negative,
                     sg=cbow_flag,
                     epochs=epochs,
                     seed=seed,
                     workers=workers)
    model.train(sentences,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    return model
