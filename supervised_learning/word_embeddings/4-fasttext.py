#!/usr/bin/env python3
"""fasttest technique"""

import gensim


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    fasttext using gensim
    """
    model = gensim.models.FastText(sentences, min_count=min_count,
                                   epochs=epochs, vector_size=vector_size,
                                   window=window, negative=negative,
                                   seed=seed, sg=cbow, workers=workers)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.epochs)

    return model
