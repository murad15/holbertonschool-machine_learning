#!/usr/bin/env python3
"""
Module containing the SARSA(lambda) reinforcement learning algorithm.
"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """
    Performs SARSA(lambda) algorithm.

    Parameters:
    - env: the gymnasium environment instance
    - Q: a numpy.ndarray of shape (s,a) containing the Q table
    - lambtha: the eligibility trace factor
    - episodes: the total number of episodes to train over
    - max_steps: the maximum number of steps per episode
    - alpha: the learning rate
    - gamma: the discount rate
    - epsilon: the initial threshold for epsilon greedy
    - min_epsilon: the minimum value that epsilon should decay to
    - epsilon_decay: the decay rate for updating epsilon between episodes

    Returns:
    - Q: the updated Q table
    """
    init_epsilon = epsilon

    for ep in range(episodes):
        # Update epsilon for the current episode using exponential decay
        epsilon = min_epsilon + (init_epsilon - min_epsilon) * \
            np.exp(-epsilon_decay * ep)

        state, _ = env.reset()
        E = np.zeros_like(Q)

        # Choose the initial action using an epsilon-greedy policy
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(Q.shape[1])
        else:
            action = np.argmax(Q[state, :])

        for _ in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Choose the next action using an epsilon-greedy policy
            if np.random.uniform(0, 1) < epsilon:
                next_action = np.random.randint(Q.shape[1])
            else:
                next_action = np.argmax(Q[next_state, :])

            # Calculate the Temporal Difference (TD) error
            delta = reward + gamma * Q[next_state, next_action] - Q[state, action]

            # Increment the eligibility trace for the visited state-action pair
            E[state, action] += 1.0

            # Update the Q-table and eligibility traces
            Q += alpha * delta * E
            E *= gamma * lambtha

            # Check if the episode is done
            if terminated or truncated:
                break

            # Move to the next step
            state = next_state
            action = next_action

    return Q
