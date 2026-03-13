#!/usr/bin/env python3
"""A module that does the trick"""

import numpy as np


def normalization_constants(X):
    """Normalization constants"""
    mean  = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
