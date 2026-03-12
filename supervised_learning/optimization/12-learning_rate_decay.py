#!/usr/bin/env python3
"""Something that function does"""


import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """Something that function does"""

    opt = tf.keras.optimizers.schedules.InverseTimeDecay(
             initial_learning_rate=alpha,
             decay_steps=decay_step,
             decay_rate=decay_rate,
             staircase=True)
    return opt
