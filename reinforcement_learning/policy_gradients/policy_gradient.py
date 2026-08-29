#!/usr/bin/env python3
"""A module to implement the policy gradient algorithm"""
import numpy as np


def policy(matrix, weight):
    """Policy function"""
    z = np.matmul(matrix, weight)

    # Softmax
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
