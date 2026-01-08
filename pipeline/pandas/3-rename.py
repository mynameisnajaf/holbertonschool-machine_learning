#!/usr/bin/env python3

"""A module to rename the columns"""


import pandas as pd


def rename(df):
    """Function that does the trick"""
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    df = df[["Datetime", "Close"]]
    return df
