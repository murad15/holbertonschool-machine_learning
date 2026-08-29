#!/usr/bin/env python3
"""
Defines the Monte Carlo algorithm for estimating a state value function.
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm to estimate a value function.

    Args:
        env: the environment instance.
        V (numpy.ndarray): array of shape (s,) containing the value
            estimate for each of the s states in the environment.
        policy (callable): function that takes in a state and returns
            the next action to take.
        episodes (int): total number of episodes to train over.
        max_steps (int): maximum number of steps per episode.
        alpha (float): learning rate.
        gamma (float): discount rate.

    Returns:
        numpy.ndarray: V, the updated value estimate.
    """
    for ep in range(episodes):
        state, _ = env.reset()
        episode = []

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(
                action)
            episode.append((state, reward))
            if terminated or truncated:
                break
            state = next_state

        G = 0
        for state, reward in reversed(episode):
            G = gamma * G + reward
            V[state] += alpha * (G - V[state])

    return V
