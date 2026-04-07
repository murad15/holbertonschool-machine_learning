#!/usr/bin/env python3
"""Something that function does"""


import tensorflow as tf


def change_contrast(image, lower, upper):
    """Something that function does"""

    contr = tf.image.random_contrast(image, lower, upper, seed=None)
    return contr
