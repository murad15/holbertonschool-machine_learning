#!/usr/bin/env python3
"""Something that function does"""


import tensorflow as tf


def l2_reg_cost(cost, model):
    """Return total cost including L2 regularization"""

    costs = []

    for i, layer in enumerate(model.layers):
        if layer.losses:
            reg = tf.add_n(layer.losses)
        else:
            reg = 0

        costs.append(cost + reg)

    return tf.stack(costs)
