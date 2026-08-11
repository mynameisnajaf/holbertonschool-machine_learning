#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def q_init(env):
    """A function that does the trick"""
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions))
    return q_table
