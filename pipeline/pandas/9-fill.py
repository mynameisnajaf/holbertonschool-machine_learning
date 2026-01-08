#!/usr/bin/env python3

"""A module that does the trick"""


def fill(df):
    """A function that does the trick"""
    df = df.drop(columns=["Weighted_Price"], errors="ignore")
    df["Close"] = df["Close"].fillna(method="ffill")
    for col in ["High", "Low", "Open"]:
        df[col] = df[col].fillna(df["Close"])
    for col in ["Volume_(BTC)", "Volume_(Currency)"]:
        df[col] = df[col].fillna(0)
    return df
