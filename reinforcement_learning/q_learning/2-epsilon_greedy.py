#!/usr/bin/env python3
"""Module for selecting an action using epsilon-greedy."""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Select the next action using the epsilon-greedy strategy."""
    p = np.random.uniform()

    if p < epsilon:
        return np.random.randint(Q.shape[1])

    return np.argmax(Q[state])
