#!/usr/bin/env python3

"""A module to load data from file"""


import pandas as pd


def from_file(filename, delimiter):
    """A function to load data from file with delimiter"""
    return pd.read_csv(filename, sep=delimiter)
