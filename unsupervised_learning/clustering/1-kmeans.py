#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """A function that does the trick"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    n, d = X.shape

    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    C = np.random.uniform(X_min, X_max, (k, d))

    for i in range(iterations):
        centroids = np.copy(C)
        centroids_2 = C[:, np.newaxis]

        dist = np.sqrt(np.sum((X - centroids_2)**2, axis=2))
        clss = np.argmin(dist, axis=0)

        for c in range(k):
            if X[clss == c].size == 0:
                C[c] = np.random.uniform(X_min, X_max, size=(1, d))
            else:
                C[c] = X[clss == c].mean(axis=0)

        centroids_2 = C[:, np.newaxis]
        dist = np.sqrt(np.sum((X - centroids_2)**2, axis=2))
        clss = np.argmin(dist, axis=0)

        if (centroids == C).all():
            break
    return C, clss
