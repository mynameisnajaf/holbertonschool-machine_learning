#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """A function that does the trick"""
    p = np.random.uniform(0, 1)
    if p < epsilon:
        next_action = np.random.randint(0, Q.shape[1])
    else:
        next_action = np.argmax(Q[state])
    return next_action
