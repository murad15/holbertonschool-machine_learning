#!/usr/bin/env python3
"""
Defines the Monte Carlo algorithm for estimating a state value function.
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Perform the Monte Carlo algorithm to estimate a value function.

    Args:
        env: The environment instance.
        V (numpy.ndarray): Value estimate for every state.
        policy (callable): Returns an action for a given state.
        episodes (int): Number of episodes.
        max_steps (int): Maximum steps per episode.
        alpha (float): Learning rate.
        gamma (float): Discount rate.

    Returns:
        numpy.ndarray: The updated value estimate.
    """
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(
                action
            )

            episode.append((state, reward))
            state = next_state

            if terminated or truncated:
                break

        returns = np.zeros(len(episode))
        G = 0

        for i in range(len(episode) - 1, -1, -1):
            G = episode[i][1] + gamma * G
            returns[i] = G

        for i in range(len(episode)):
            state = episode[i][0]
            V[state] += alpha * (returns[i] - V[state])

    return V
