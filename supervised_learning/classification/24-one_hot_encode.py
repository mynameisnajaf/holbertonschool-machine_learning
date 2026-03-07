#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np

def one_hot_encode(Y, classes):
    """One-hot encode Y"""
    ohe = np.zeros((classes, len(Y)))
    ohe[Y, np.arange(len(Y))] = 1
    return ohe
