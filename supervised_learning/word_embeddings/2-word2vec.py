#!/usr/bin/env python3
"""A module that creates a Word2Vec model"""

import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Create and train a Word2Vec model."""

    if cbow is True:
        cbow_flag = 0
    else:
        cbow_flag = 1
    model = gensim.models.Word2Vec(sentences=sentences,
                     min_count=min_count,
                     window=window,
                     negative=negative,
                     sg=cbow_flag,
                     seed=seed,
                     workers=workers)
    model.train(sentences,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    return model
