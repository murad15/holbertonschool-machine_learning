#!/usr/bin/env python3
"""This function does something interesting""""


import tensorflow as tf


def change_brightness(image, max_delta):
    """This function does something interesting""""

    bright = tf.image.adjust_brightness(image, max_delta)
    return bright
