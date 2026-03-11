#!/usr/bin/env python3
"""A module for supervised learning"""

import tensorflow.keras as K
import numpy as np


def one_hot(labels, classes=None):
    """One-hot encoding of labels"""
    if len(labels) == 0:
        return None
    elif type(classes) is not int:
        return None
    elif not isinstance(labels, np.ndarray):
        return None
    elif classes <= np.amax(labels):
        return None

    ohe = np.zeros((classes, len(labels)))
    ohe[labels, np.arange(len(labels))] = 1
    return ohe
