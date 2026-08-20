#!/usr/bin/env python3
"""
TD(λ) Value Estimation for Reinforcement Learning.
"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    This function performs the TD(λ) algorithm to estimate the value function.
    """
    for _ in range(episodes):
        E = np.zeros_like(V)
        state = env.reset()[0]

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            delta = reward + gamma * V[next_state] - V[state]
            E[state] += 1
            V += alpha * delta * E
            E *= gamma * lambtha
            state = next_state

            if terminated or truncated:
                break

    return V
