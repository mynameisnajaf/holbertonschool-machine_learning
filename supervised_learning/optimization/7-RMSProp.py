#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Updates the RMSProp variable"""
    vt = beta2 * s + (1 - beta2) * grad ** 2
    rms = var - alpha * grad / (np.sqrt(vt + epsilon))
    return rms, vt
