#!/usr/bin/env python3
"""A module to implement the policy gradient algorithm"""
import numpy as np


def policy(matrix, weight):
    """Policy function"""
    z = np.matmul(matrix, weight)

    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def policy_gradient(state, weight):
    """Computes the Monte-Carlo policy gradient based on a state and weight matrix."""
    state = np.reshape(state, (1, -1))

    p = policy(state, weight)

    action = np.random.multinomial(1, p[0]).argmax()

    one_hot = np.zeros(p.shape[1])
    one_hot[action] = 1

    gradient = np.outer(state[0], one_hot - p[0])

    return action, gradient
