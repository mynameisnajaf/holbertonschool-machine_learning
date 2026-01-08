#!/usr/bin/env python3

"""A module that does the trick"""


import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """A function that does the trick"""
    df1 = index(df1)
    df2 = index(df2)
    df = pd.concat(
        [
            df2.loc[1417411980:1417417980],
            df1.loc[1417411980:1417417980]
        ],
        keys=["bitstamp", "coinbase"]
    )
    df = df.swaplevel(0, 1, axis=0)
    return df.sort_index()
