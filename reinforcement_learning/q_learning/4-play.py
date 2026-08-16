#!/usr/bin/env python3
"""Module for loading a FrozenLake environment."""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Load and return an unwrapped FrozenLake environment."""
    return gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode="ansi"
    ).unwrapped
