#!/usr/bin/env python3
"""Monte Carlo reinforcement learning algorithm."""


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Update a state-value estimate using every-visit Monte Carlo.

    Args:
        env: Environment instance.
        V: Value estimate for each state.
        policy: Function that returns an action for a given state.
        episodes: Number of training episodes.
        max_steps: Maximum steps allowed per episode.
        alpha: Learning rate.
        gamma: Discount rate.

    Returns:
        The updated value estimate.
    """
    for _ in range(episodes):
        state = env.reset()[0]
        episode = []

        for _ in range(max_steps):
            action = policy(state)

            next_state, reward, terminated, truncated, _ = env.step(action)

            episode.append((state, reward))
            state = next_state

            if terminated or truncated:
                break

        G = 0

        for state, reward in reversed(episode):
            G = reward + gamma * G
            V[state] += alpha * (G - V[state])

    return V
