#!/usr/bin/env python3

"""A module to slice the data"""


def slice(df):
    """A function that does the trick"""
    columns = ["High", "Low", "Close", "Volume_BTC"]
    df_selected = df[columns]
    return df_selected.iloc[::60]
