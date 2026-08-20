#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """Monte Carlo algorithm"""
    for _ in range(episodes):
        episode = []
        state, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), None)

        for _ in range(max_steps):
            action = policy(state)
            step_result = env.step(action)

            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_result

            episode.append((state, reward))
            state = next_state

            if done:
                break

        G = 0
        visited_states = set()

        for state, reward in reversed(episode):
            G = gamma * G + reward

            if state not in visited_states:
                visited_states.add(state)
                V[state] = V[state] + alpha * (G - V[state])

    return V
