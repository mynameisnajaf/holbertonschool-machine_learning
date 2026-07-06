#!/usr/bin/env python3
"""Positional encoding"""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """calculates the positional encoding for a transformer"""
    positional_encoding = np.zeros([max_seq_len, dm])

    for pos in range(max_seq_len):
        for i in range(0, dm, 2):
            positional_encoding[pos, i] = np.sin(get_angles(pos, i, dm))
            positional_encoding[pos, i + 1] = np.cos(get_angles(pos, i, dm))
    return positional_encoding


def get_angles(pos, i, dm):
    """calculates the angles for a transformer"""
    angle_rates = 1 / (10000 ** (i / dm))
    return pos * angle_rates
