#!/usr/bin/env python3
"""A module to implement the policy gradient algorithm"""
import numpy as np


def policy(matrix, weight):
    """Policy function"""
    z = np.matmul(matrix, weight)

    # Softmax
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def policy_gradient(state, weight):
    """Computes the Monte-Carlo policy gradient based on a state and weight matrix."""
    Policy = policy(state, weight)
    action = np.random.choice(len(Policy[0]), p=Policy[0])
    s = Policy.reshape(-1, 1)
    softmax = (np.diagflat(s) - np.dot(s, s.T))[action, :]
    dlog = softmax / Policy[0, action]
    gradient = state.T.dot(dlog[None, :])
    return action, gradient
