#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def initialize(X, k):
    """A function that does the trick"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    return np.random.uniform(X_min, X_max, (k, d))
