#!/usr/bin/env python3
"""A module that does the trick"""

import numpy as np


def normalize(X, m, s):
    """Normalize the data"""
    return (X - m) / s
