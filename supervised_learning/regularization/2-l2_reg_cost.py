#!/usr/bin/env python3
"""Something that function does"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """Return total cost including L2 regularization"""

    reg_cost = tf.add_n([layer.losses for i, layer in enumerate(
        model.layers) if layer.losses])

    return cost + reg_cost
