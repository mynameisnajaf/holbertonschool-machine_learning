#!/usr/bin/env python3
"""A module that does the trick"""
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """A function that does the trick"""
    linkage = sch.linkage(X, method='ward')
    clss = sch.fcluster(linkage, t=dist, criterion='distance')
    plt.figure()
    sch.dendrogram(linkage, color_threshold=dist)
    plt.show()
    return clss
