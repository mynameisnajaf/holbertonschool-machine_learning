#!/usr/bin/env python3

"""A module to change values"""


import numpy as np
import pandas as pd


def array(df):
    """Convert these selected values into a
    numpy.ndarray."""
    return df.tail(10)[['High', 'Close']].to_numpy()
