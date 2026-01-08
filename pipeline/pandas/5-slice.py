#!/usr/bin/env python3

"""A module to slice the data"""


def slice(df):
    """A function that does the trick"""
    return df.loc[60, ["High", "Low", "Close", "Volume_BTC"]]
