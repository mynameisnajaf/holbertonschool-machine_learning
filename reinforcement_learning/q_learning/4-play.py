#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def play(env, Q, max_steps=100):
    """A function that does the trick"""
    rendered_outputs = []
    total_reward = 0.0
    state, _ = env.reset()
    rendered_outputs.append(env.render())

    for step in range(max_steps):
        action = np.argmax(Q[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        rendered_outputs.append(env.render())
        state = next_state

        if terminated or truncated:
            break

    return total_reward, rendered_outputs
