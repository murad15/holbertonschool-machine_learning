#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Something that function does"""

    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    output_h = h
    output_w = w

    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            window = padded[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return output
