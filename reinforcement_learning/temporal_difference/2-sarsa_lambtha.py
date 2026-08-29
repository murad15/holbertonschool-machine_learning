#!/usr/bin/env python3
"""
Defines the SARSA(lambda) reinforcement learning algorithm.
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Select an action using the epsilon-greedy strategy.

    Args:
        Q (numpy.ndarray): The Q table.
        state (int): The current state.
        epsilon (float): Exploration threshold.

    Returns:
        int: The selected action.
    """
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(Q.shape[1])

    return np.argmax(Q[state])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1,
                  min_epsilon=0.1, epsilon_decay=0.05):
    """
    Perform the SARSA(lambda) algorithm.

    Args:
        env: The environment instance.
        Q (numpy.ndarray): The Q table.
        lambtha (float): Eligibility trace factor.
        episodes (int): Number of training episodes.
        max_steps (int): Maximum steps per episode.
        alpha (float): Learning rate.
        gamma (float): Discount rate.
        epsilon (float): Initial exploration threshold.
        min_epsilon (float): Minimum exploration threshold.
        epsilon_decay (float): Epsilon decay rate.

    Returns:
        numpy.ndarray: The updated Q table.
    """
    initial_epsilon = epsilon
    eligibility = np.zeros_like(Q)

    for episode in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon)

        for _ in range(max_steps):
            eligibility *= lambtha * gamma
            eligibility[state, action] += 1

            next_state, reward, terminated, truncated, _ = env.step(
                action
            )

            next_action = epsilon_greedy(
                Q, next_state, epsilon
            )

            delta = (
                reward
                + gamma * Q[next_state, next_action]
                - Q[state, action]
            )

            Q[state, action] += (
                alpha * delta * eligibility[state, action]
            )

            if terminated or truncated:
                break

            state = next_state
            action = next_action

        epsilon = min_epsilon + (
            (initial_epsilon - min_epsilon)
            * np.exp(-epsilon_decay * episode)
        )

    return Q
