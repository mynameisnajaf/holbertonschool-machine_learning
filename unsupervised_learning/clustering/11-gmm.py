#!/usr/bin/env python3
"""A module that does the trick"""
import sklearn.mixture


def gmm(X, k):
    """A function that does the trick"""
    g = sklearn.mixture.GaussianMixture(n_components=k)
    g.fit(X)
    pi = g.weights_
    m = g.means_
    S = g.covariances_
    clss = g.predict(X)
    bic = g.bic(X)
    return pi, m, S, clss, bic
