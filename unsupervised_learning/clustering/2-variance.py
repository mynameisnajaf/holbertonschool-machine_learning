#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def variance(X, C):
    """A function that does the trick"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    n, d = X.shape
    centroids_2 = C[:, np.newaxis]
    dist = np.sqrt(np.sum((X - centroids_2) ** 2, axis=2))
    min_dist = np.min(dist, axis=0)

    variance = np.sum(min_dist ** 2)
    return variance