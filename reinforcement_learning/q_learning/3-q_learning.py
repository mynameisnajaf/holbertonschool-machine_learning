#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """A function that does the trick"""
    total_rewards = []
    n_cols = len(env.unwrapped.desc[0])

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)

            row = next_state // n_cols
            col = next_state % n_cols
            if env.unwrapped.desc[row][col] == b'H':
                reward = -1
            Q[state, action] = Q[state, action] + alpha * (
                    reward + gamma * np.max(Q[next_state]) - Q[state, action])
            episode_reward += reward
            state = next_state
            if terminated or truncated:
                break
        total_rewards.append(episode_reward)

        epsilon = max(min_epsilon, epsilon - epsilon_decay)

    return Q, total_rewards
