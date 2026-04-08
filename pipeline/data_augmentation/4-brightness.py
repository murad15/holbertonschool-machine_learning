#!/usr/bin/env python3
"""Randomly change image brightness using TensorFlow."""

import tensorflow as tf


def change_brightness(image, max_delta):
    """Randomly changes the brightness of an image.

    Args:
        image: A 3D tf.Tensor containing the image.
        max_delta: Maximum amount to brighten or darken the image.

    Returns:
        The altered image as a tf.Tensor.
    """
    return tf.image.random_brightness(image, max_delta)
