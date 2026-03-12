#!/usr/bin/env python3
"""Something that function does"""


import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """Something that function does"""

    opt = tf.keras.optimizers.RMSprop(
             learning_rate=alpha,
             rho=beta2,
             epsilon=epsilon)
    return opt
