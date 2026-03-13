#!/usr/bin/env python3
"""A module that does the trick"""

import numpy as np


def shuffle_data(X, Y):
    """Shuffles the data"""
    m = X.shape[0]
    shuffle = np.random.permutation(m)
    X = X[shuffle]
    Y = Y[shuffle]
    return X, Y
