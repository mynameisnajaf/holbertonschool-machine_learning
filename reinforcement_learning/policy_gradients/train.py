#!/usr/bin/env python3
"""A module to implement the policy gradient algorithm"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """The main function that implements the policy gradient algorithm"""
    weight = np.random.rand(env.observation_space.shape[0],
                            env.action_space.n)

    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        state = np.reshape(state, (1, -1))

        grads = []
        rewards = []
        score = 0

        done = False

        while not done:
            action, grad = policy_gradient(state, weight)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            grads.append(grad)
            rewards.append(reward)

            score += reward
            state = np.reshape(next_state, (1, -1))

            if show_result and episode % 1000 == 0:
                env.render()

        discounted = np.zeros(len(rewards))
        running = 0

        for t in range(len(rewards) - 1, -1, -1):
            running = rewards[t] + gamma * running
            discounted[t] = running

        for grad, reward in zip(grads, discounted):
            weight += alpha * grad * reward

        scores.append(score)

        print("Episode: {} Score: {}".format(episode, score))

    return scores
