#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """A function that does the trick"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k < 1:
        return None, None, None
    n, d = X.shape
    phi = np.ones(k) / k
    m, _ = kmeans(X, k)
    S = np.tile(np.identity(d), (k, 1)).reshape(k, d, d)
    return phi, m, S
