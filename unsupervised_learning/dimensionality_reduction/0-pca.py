#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def pca(X, var=0.95):
    """A function that does the trick"""
    U, sigma, V = np.linalg.svd(X)
    explained_variance = (sigma ** 2) / np.sum(sigma ** 2)
    cumulative_variance = np.cumsum(explained_variance)
    nd = np.where(cumulative_variance >= var)[0][0] + 1
    W = V.T[:, :nd]

    return W
