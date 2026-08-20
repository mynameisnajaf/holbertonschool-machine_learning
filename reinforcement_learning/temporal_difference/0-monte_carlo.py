#!/usr/bin/env python3
"""Monte Carlo algorithm."""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Performs the Monte Carlo algorithm."""
    for ep in range(episodes):
        state = env.reset()[0]
        episode = []

        # Generate an episode
        for step in range(max_steps):
            action = policy(state)
            next_state, reward, done, truncated, info = env.step(action)
            episode.append((state, reward))
            state = next_state
            if done or truncated:
                break

        episode = np.array(episode, dtype=int)

        # Compute returns and update V (first-visit MC)
        G = 0
        for t in reversed(range(len(episode))):
            state_t, reward_t = episode[t]
            G = reward_t + gamma * G

            # First-visit check
            if state_t not in episode[:t, 0]:
                V[state_t] = V[state_t] + alpha * (G - V[state_t])

    return V
