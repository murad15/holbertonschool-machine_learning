#!/usr/bin/env python3
"""This function does something interesting"""

import tensorflow as tf


def change_hue(image, max_delta):
    """This function does something interesting""""

    hue = tf.image.adjust_hue(image, delta, name=None)
    return hue
