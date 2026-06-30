#!/usr/bin/env python3
"""A module that creates a Word2Vec model"""

import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Create and train a Word2Vec model."""

    model = gensim.models.Word2Vec(sentences, min_count=min_count,
                                   epochs=epochs, vector_size=vector_size,
                                   window=window, negative=negative,
                                   seed=seed, sg=cbow, workers=workers)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.epochs)

    return model
