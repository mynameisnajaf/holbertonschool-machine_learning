#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np
from sklearn.covariance import log_likelihood

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """A function that does the trick"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None

    n, d = X.shape
    k = pi.shape[0]
    if d != m.shape[1] or d != S.shape[1] or d != S.shape[2]:
        return None, None
    if k != m.shape[0] or k != S.shape[0]:
        return None, None

    mean_centroids = m
    covarience_matrix = S
    gauss = np.zeros((k, n))
    for i in range(k):
        likelihood = pdf(X, mean_centroids[i], covarience_matrix[i])
        prior = pi[i]
        gauss[i] = likelihood * prior
    gauss_last = gauss / np.sum(gauss, axis=0)
    log_likelihood = np.sum(np.log(np.sum(gauss, axis=0)))
    return gauss_last, log_likelihood
