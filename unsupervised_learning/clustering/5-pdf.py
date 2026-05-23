#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def pdf(X, m, S):
    """A function that does the trick"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None
    n, d = X.shape
    mean = m
    x_m = X - mean
    det_S = np.linalg.det(S)
    inv_S = np.linalg.inv(S)

    part_1_dem = np.sqrt(det_S) * ((2 * np.pi) ** (d/2))
    part_2 = np.matmul(x_m, inv_S)
    part_2_1 = np.sum(x_m * part_2, axis=1)
    part_2_2 = np.exp(part_2_1 / -2)
    pdf = part_2_2 / part_1_dem
    P = np.where(pdf < 1e-300, 1e-300, pdf)
    return P
