#!/usr/bin/env python3

"""Flip and switch"""


def flip_switch(df):
    """A function that does the trick"""
    df = df.sort_index(axis=1, ascending=True)
    return df.T
