#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Updates the RMSProp variable"""
    vt = beta2 * var + (1 - beta2) * grad ** 2
    var = s - alpha * grad / (np.sqrt(vt + epsilon))
    return var
