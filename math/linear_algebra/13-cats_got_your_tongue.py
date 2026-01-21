#!/usr/bin/env python3

"""A module that does the trick"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """A function that does the trick"""
    return np.concatenate((mat1, mat2), axis=axis)
