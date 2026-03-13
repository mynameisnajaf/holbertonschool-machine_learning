#!/usr/bin/env python3
"""A module that does the trick"""

import numpy as np


def shuffle_data(X, Y):
    """Shuffles the data"""
    return np.random.permutation(X), np.random.permutation(Y)
