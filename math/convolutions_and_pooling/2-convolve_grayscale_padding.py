#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Something that function does"""

    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    output_h = h - kh + 1 + 2 * ph
    output_w = w - kw + 1 + 2 * pw

    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            window = images[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return output


