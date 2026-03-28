#!/usr/bin/env python3
"""Something that function does"""


import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Something that function does"""

    m, h, w = images.shape
    kh, kw = kernel.shape
    if padding == 'same':
        ph, pw = kh // 2, kw // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    sh, sw = stride

    output_h = int(np.floor((h + 2*ph - kh) / sh) + 1)
    output_w = int(np.floor((w + 2*pw - kw) / sw) + 1)

    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            window = padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw]
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return output
