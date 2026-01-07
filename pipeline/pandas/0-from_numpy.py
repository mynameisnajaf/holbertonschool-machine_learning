#!/usr/bin/env python3
"""
0-from_numpy module.

This module contains the from_numpy function, which converts a
NumPy array into a pandas DataFrame with columns labeled from
'A' to 'Z', matching the number of columns in the array.
"""

import pandas as pd


def from_numpy(array):
    """A function to create a dataframe"""
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return pd.DataFrame(array, columns=alphabet[:array.shape[1]])
