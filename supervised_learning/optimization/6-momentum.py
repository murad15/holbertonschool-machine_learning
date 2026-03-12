#!/usr/bin/env python3
"""Something that function does"""


import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """Something that function does"""

    optimizer = tf.keras.optimizers.SGD(
                learning_rate=alpha,
                momentum=beta1)
    return optimizer
