#!/usr/bin/env python3
"""Module for initializing a Q-table."""

import numpy as np


def q_init(env):
    """Initialize and return a Q-table filled with zeros."""
    return np.zeros((env.observation_space.n, env.action_space.n))
