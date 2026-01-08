#!/usr/bin/env python3

"""Flip and switch"""


def flip_switch(df):
    """A function that does the trick"""
    df = df.sort_index(ascending=False)
    return df.T
