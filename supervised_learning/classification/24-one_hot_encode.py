#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def one_hot_encode(Y, classes):
    """One-hot encode Y"""
    if len(Y) == 0:
        return None
    elif type(classes) is not int:
        return None
    elif not isinstance(Y, np.ndarray):
        return None
    elif classes <= np.amax(Y):
        return None

    ohe = np.zeros((classes, len(Y)))
    ohe[Y, np.arange(len(Y))] = 1
    return ohe
