#!/usr/bin/env python3
"""
Defines the TD(lambda) algorithm for estimating a state value function.
"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """
    Perform the TD(lambda) algorithm to estimate a value function.

    Args:
        env: The environment instance.
        V (numpy.ndarray): Value estimate for every state.
        policy (callable): Returns an action for a given state.
        lambtha (float): Eligibility trace factor.
        episodes (int): Number of training episodes.
        max_steps (int): Maximum steps per episode.
        alpha (float): Learning rate.
        gamma (float): Discount rate.

    Returns:
        numpy.ndarray: The updated value estimate.
    """
    for _ in range(episodes):
        state, _ = env.reset()
        eligibility = np.zeros_like(V)

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(
                action
            )

            delta = reward + gamma * V[next_state] - V[state]
            eligibility[state] += 1

            V += alpha * delta * eligibility
            eligibility *= gamma * lambtha

            if terminated or truncated:
                break

            state = next_state

    return V
