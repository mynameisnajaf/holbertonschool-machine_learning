#!/usr/bin/env python3
"""A module to implement regularization cost"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """L2 regularization cost with gradient descent"""
    m = Y.shape[1]
    W = weights.copy()

    for i in reversed(range(L)):
        A = cache['A' + str(i + 1)]
        if L - 1 == i:
            dZ = cache['A' + str(i + 1)] - Y
            dW = (np.matmul(cache['A' + str(i)], dZ.T) / m).T
            dW_L2 = dW + (lambtha / m) * W['W' + str(i + 1)]
            db = np.sum(dZ, axis=1, keepdims=True) / m

        else:
            dW2 = np.matmul(W['W' + str(i + 1)].T, dZ2) / m
            tanh = 1 - (A * A)
            dZ = dW2 * tanh
            dW = np.matmul(dZ, cache["A" + str(i)].T) / m
            dW_L2 = dW + (lambtha / m) * W["W" + str(i + 1)]
            db = np.sum(dZ, axis=1, keepdims=True) / m
        weights["W" + str(i + 1)] = (W["W" + str(i + 1)] - (alpha * dW_L2))
        weights["b" + str(i + 1)] = W["b" + str(i + 1)] - (alpha * db)
        dZ2 = dZ
