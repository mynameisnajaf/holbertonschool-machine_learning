#!/usr/bin/env python3

"""A module that does the trick"""


def analyze(df):
    """A function that does the trick"""
    return df.drop(columns=["Timestamp"]).describe()
