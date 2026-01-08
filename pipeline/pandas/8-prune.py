#!/usr/bin/env python3

"""A module that does the trick"""


def prune(df):
    """A function that does the trick"""
    return df.dropna(subset=["Close"], how="any")
