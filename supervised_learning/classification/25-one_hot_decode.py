#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def one_hot_decode(one_hot):
    """One-hot decode"""
    if not isinstance(one_hot, np.ndarray):
        return None
    elif len(one_hot) == 0:
        return None
    elif len(one_hot.shape) != 2:
        return None
    elif not np.all((one_hot == 0) | (one_hot == 1)):
        return None
    return np.argmax(one_hot, axis=0)
