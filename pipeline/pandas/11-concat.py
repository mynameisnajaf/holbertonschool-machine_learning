#!/usr/bin/env python3

"""A module that does the trick"""


import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """A function that does the trick"""
    df1 = index(df1)
    df2 = index(df2)
    df2_selected = df2.loc[:1417411920]
    return pd.concat([df2_selected, df1], keys=["bitstamp", "coinbase"])
