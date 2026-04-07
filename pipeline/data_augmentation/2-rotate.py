#!/usr/bin/env python3
"""Something that function does"""


import tensorflow as tf


def rotate_image(image):
    """Something that function does"""

    rot = tf.image.rot90(image, k=1, name=None)
    return rot
