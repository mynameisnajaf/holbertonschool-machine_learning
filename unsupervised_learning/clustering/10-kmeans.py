#!/usr/bin/env python3
"""A module that does the trick"""
import sklearn.cluster


def kmeans(X, k):
    """A function that does the trick"""
    k_means = sklearn.cluster.KMeans    (n_clusters=k)
    k_means.fit(X)
    clss = k_means.labels_
    C = k_means.cluster_centers_
    return clss, C
