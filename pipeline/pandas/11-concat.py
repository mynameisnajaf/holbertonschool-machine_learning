#!/usr/bin/env python3

"""A module that does the trick"""


import pandas as pd
index = __import__(&#39;10-index&#39;).index


def concat(df1, df2):
    """A function that does the trick"""
    df1 = index(df1)
    df2 = index(df2)
    return pd.concat([df2.loc[:1417411920], df1], keys=["bitstamp", "coinbase"])
