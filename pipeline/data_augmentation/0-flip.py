#!/usr/bin/env python3
"""Something that function does"""


import tensorflow as tf


def flip_image(image):
    """Something that function does"""

    flipped = tf.image.flip_left_right(image)
    return flipped
