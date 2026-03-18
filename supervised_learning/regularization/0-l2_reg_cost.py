#!/usr/bin/env python3
"""A module to implement regularization cost"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """L2 regularization cost"""
    normal = 0
    for w, b in weights.items():
        if w[0] == 'W':
            normal = normal + np.linalg.norm(b)
    l2_cost = cost + (lambtha * normal / (2 * m))
    return l2_cost
