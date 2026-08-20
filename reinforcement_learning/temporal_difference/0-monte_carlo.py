#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Performs Monte Carlo value estimation."""

    for _ in range(episodes):
        state = env.reset()

        # Handle Gym/Gymnasium reset format
        if isinstance(state, tuple):
            state = state[0]

        episode = []

        for _ in range(max_steps):
            action = policy(state)
            result = env.step(action)

            # Handle both old and new Gym APIs
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            episode.append((state, reward))
            state = next_state

            if done:
                break

        # Calculate returns backwards
        G = 0
        visited = set()

        for state, reward in reversed(episode):
            G = gamma * G + reward

            # First-visit Monte Carlo
            if state not in visited:
                V[state] += alpha * (G - V[state])
                visited.add(state)

    return V
