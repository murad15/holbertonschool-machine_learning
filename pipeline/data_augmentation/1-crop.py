#!/usr/bin/env python3
"""Something that function does"""


import tensorflow as tf


def crop_image(image, size):
    """Something that function does"""

    cropped = tf.image.random_crop(image, size)
    return cropped
