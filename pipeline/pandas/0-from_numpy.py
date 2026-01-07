#!/usr/bin/env python3

import pandas as pd


"""A module that contains function"""

def from_numpy(array):
    """A function to create a dataframe"""
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return pd.DataFrame(array, columns=alphabet[:array.shape[1]])
