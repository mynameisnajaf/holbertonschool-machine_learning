#!/usr/bin/env python3
"""A module for supervised learning"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """One-hot encoding of labels"""

    ohe = np.zeros((classes, len(labels)))
    ohe[labels, np.arange(len(labels))] = 1
    return ohe
