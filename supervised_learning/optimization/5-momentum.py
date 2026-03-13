#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Updates the momentum"""
    vt = beta1 * v + (1 - beta1) * grad
    w = var - alpha * vt
    return w, vt
