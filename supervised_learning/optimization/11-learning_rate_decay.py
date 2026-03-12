#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Something that function does"""

    lr = alpha / (1 + decay_rate * np.floor(global_step / decay_step))

    return lr
