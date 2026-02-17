#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """A function that does the trick"""
    S = sensitivity(confusion)
    P = precision(confusion)
    f1 = 2 * P * S / (P + S)
    return f1
