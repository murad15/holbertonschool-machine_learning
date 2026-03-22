#!/usr/bin/env python3
"""Something that function does"""


import tensorflow as tf


def l2_reg_cost(cost, model):
    """Return total cost including L2 regularization"""

    l2_penalties = tf.math.add_n(model.losses)

    total_cost = cost + l2_penalties
    return total_cos
