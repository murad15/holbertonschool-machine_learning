#!/usr/bin/env python3
"""Something that function does"""

import numpy as np

class Neuron:
    """Something that function does"""

    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Weights initialized with random normal distribution
        self.W = np.random.randn(1, nx)

        # Bias initialized to 0
        self.b = 0

        # Activated output initialized to 0
        self.A = 0
